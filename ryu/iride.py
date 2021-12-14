# =============================================================================
#
#    intelligent Routing of IDentified Elephants (iRIDE)
#
# =============================================================================

# https://ryu.readthedocs.io/en/latest/ofproto_v1_3_ref.html
# https://thispointer.com/python-how-to-create-a-thread-to-run-a-function-in-parallel/
# https://www.bogotobogo.com/python/Multithread/python_multithreading_Synchronization_Lock_Objects_Acquire_Release.php

__title__ = "intelligent Routing of IDentified Elephants (iRIDE)"
__author__ = "festradasolano"
__year__ = "2021"

import copy
import numpy as np
import sys
import time
import tensorflow as tf
import threading

# from .pm2 import PM2Routing  # TODO uncomment for developing
from pm2 import PM2Routing  # TODO uncomment for running

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, ipv4, udp
from ryu.lib.packet import ether_types, in_proto

# ==============================
# Main API Class (OpenFlow v1.3)
# ==============================


class IRide(PM2Routing):
    FATTREE_LEN_PATH_EDGE = 5
    FATTREE_LEN_PATH_POD_EDGE = 3

    # -------------------------------------------------------------------------
    # TODO SET Constants for each test run
    # -------------------------------------------------------------------------
    BATCH_SIZE = 1  # samples
    MONITOR_RATE_EFLOW = 5  # seconds
    EFLOW_SIZE_THRESHOLD = 1_000  # bytes TODO to 100_000
    EFLOW_RATE_DIVISOR = 1_000  # to Kbps
    COLD_START_THRESHOLD = 4  # trains
    MIN_RATE = 0.01  # Kbps
    MAX_RATE = 1_000  # Kbps
    MIN_DURATION = 0.01  # seconds
    MAX_DURATION = 400  # seconds
    DEFAULT_DURATION = 10  # seconds
    UNIT_RATE = "Kbps"

    # TODO comment/uncomment univ1
    HIDDEN_LAYERS_RATE = 5
    HIDDEN_UNITS_RATE = 480
    LAMBDA_RATE = 0.1
    DROPOUT_RATE = 0.5
    HIDDEN_LAYERS_TIME = 7
    HIDDEN_UNITS_TIME = 60
    LAMBDA_TIME = 0
    DROPOUT_TIME = 0

    # TODO comment/uncomment univ2
    # HIDDEN_LAYERS_RATE = 8
    # HIDDEN_UNITS_RATE = 600
    # LAMBDA_RATE = 0.0001
    # DROPOUT_RATE = 0.25
    # HIDDEN_LAYERS_TIME = 5
    # HIDDEN_UNITS_TIME = 60
    # LAMBDA_TIME = 0
    # DROPOUT_TIME = 0

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    INPUT_UNITS = 117

    TOS_EFLOW_REPORT = 60
    DSCP_EFLOW_REPORT = 15
    PORT_MISSED_TCP_UDP = 0
    PRIOR_EFLOW_REPORT = 65535
    PRIOR_EFLOW = 200
    IDLETO_EFLOW = 5
    COOKIE_EFLOW = DSCP_EFLOW_REPORT

    FIRST_N_PKTS = 7
    BYTES_PKT_SIZE = 2
    BYTES_PKT_IAT = 4
    NUMBER_IP_OCTETS = 4
    BITS_IN_IP_OCTET = 8
    BITS_IN_BYTE = 8
    USEC_IN_SEC = 1_000_000

    LEN_PROTOS_EFLOW_REPORT = 4
    LEN_PAYLOAD_EFLOW_REPORT = (1 + (FIRST_N_PKTS * BYTES_PKT_SIZE)
                                + ((FIRST_N_PKTS - 1) * BYTES_PKT_IAT))

    # -------------------------------------------------------------------------
    # Initialization Process
    # -------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super(IRide, self).__init__(*args, **kwargs)

        # Initialize dictionaries
        self.eflows = dict()
        self.edge_paths = dict()

        # Initialize tensors and counters
        self.train_count = 0
        self.x_train = tf.convert_to_tensor(
            np.empty((0, self.INPUT_UNITS), np.float64)
        )
        self.y_train_rate = tf.convert_to_tensor(
            np.empty((0, 1), np.float64)
        )
        self.y_train_time = tf.convert_to_tensor(
            np.empty((0, 1), np.float64)
        )

        # Build neural network models for rate and duration
        self.logger.info("Building ML models for rate and duration")
        self.model_rate, self.cpmodel_rate = IRide.build_tensorflow_model(
            input_units=self.INPUT_UNITS,
            hidden_layers=self.HIDDEN_LAYERS_RATE,
            hidden_units=self.HIDDEN_UNITS_RATE,
            lmbda=self.LAMBDA_RATE,
            dropout=self.DROPOUT_RATE
        )
        self.model_time, self.cpmodel_time = IRide.build_tensorflow_model(
            input_units=self.INPUT_UNITS,
            hidden_layers=self.HIDDEN_LAYERS_TIME,
            hidden_units=self.HIDDEN_UNITS_TIME,
            lmbda=self.LAMBDA_TIME,
            dropout=self.DROPOUT_TIME
        )

        # Initialize threading locks
        self.lock_models = threading.Lock()
        self.lock_eflows = threading.Lock()

    # -------------------------------------------------------------------------
    # Overriden: add the PM2 group and flow rules to Edge switch and add flow
    # rule to sent elephant report packets to the controller
    # -------------------------------------------------------------------------
    def add_edge_rules(self, dp):
        # Add PM2 group and flow rules
        super(IRide, self).add_edge_rules(dp)

        # Add flow rule: elephant report -> controller
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                    ip_dscp=self.DSCP_EFLOW_REPORT,
                                    ip_proto=in_proto.IPPROTO_UDP)
        actions = [ofp_parser.OFPActionOutput(ofp.OFPP_CONTROLLER)]
        self.add_flow(dp, self.PRIOR_EFLOW_REPORT, match, actions)
        self.logger.info(
            "  Switch (%s) flow rule: "
            + "Table:0 Priority:%d [IP, DSCP(%d), UDP] -> CONTROLLER",
            self.get_dpid_string(dp.id), self.PRIOR_EFLOW_REPORT,
            self.DSCP_EFLOW_REPORT
        )

    # -------------------------------------------------------------------------
    # Overriden RYU event: packet-in handler. Call the PM2 packet-in handler
    # for handling LLDP and ARP packets, and add IPv4 packets handler
    # -------------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        # Check if PM2 handled LLDP and ARP packets
        if super(IRide, self).packet_in_handler(ev):
            return True

        # Get Message Data
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)

        # Handle IPv4 packets
        if eth_pkt.ethertype == ether_types.ETH_TYPE_IP:
            return self.ipv4_packet_handler(msg, pkt)

        # No more packets to handle
        return False

    # -------------------------------------------------------------------------
    # IPv4 packet handler
    # -------------------------------------------------------------------------
    def ipv4_packet_handler(self, msg, pkt):
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        # Handle elephant flow report IPv4 packet
        if (ip_pkt.tos == self.TOS_EFLOW_REPORT
                and ip_pkt.proto == in_proto.IPPROTO_UDP):
            self.eflow_report_packet_handler(msg, pkt)
            return True

        # No more IPv4 packets to handle
        return False

    # -------------------------------------------------------------------------
    # Elephant flow report packet handler
    # -------------------------------------------------------------------------
    def eflow_report_packet_handler(self, msg, pkt):
        dp = msg.datapath
        in_port = msg.match["in_port"]
        self.logger.info(
            "\nReceived elephant flow report packet in Switch (%s), port %s",
            self.get_dpid_string(dp.id), in_port
        )

        # Check packet has the correct number of protocols
        if len(pkt.protocols) != self.LEN_PROTOS_EFLOW_REPORT:
            self.logger.warning("  Packet with wrong length of protocols")
            return

        # Get payload and check size
        payload = pkt.protocols[-1]
        if len(payload) != self.LEN_PAYLOAD_EFLOW_REPORT:
            self.logger.warning("  Packet with wrong payload length")
            return

        # Get IP data
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        ip_src = ip_pkt.src
        ip_dst = ip_pkt.dst

        # Check both source and destination IPs are in routing table
        if ip_src not in self.ip_table:
            self.logger.warning(
                "  Reported source IP (%s) not in IP table", ip_src
            )
            return
        elif ip_dst not in self.ip_table:
            self.logger.warning(
                "  Reported destination IP (%s) not in IP table", ip_dst
            )
            return

        # Check if source and destination hosts are in the same switch
        src = self.ip_table[ip_src]
        dpid_src = src["dp"].id
        dst = self.ip_table[ip_dst]
        dpid_dst = dst["dp"].id
        if dpid_src == dpid_dst:
            # No need to install flow rule; only one path exists
            self.logget.info(
                "  Source (%s) and destination (%s) hosts are connected to "
                + "the same Edge switch -> no multipath availability",
                ip_src, ip_dst
            )
            return

        # Get UDP data
        udp_pkt = pkt.get_protocol(udp.udp)
        port_src = udp_pkt.src_port
        port_dst = udp_pkt.dst_port

        # Parse payload: IP protocol and first N packet sizes and IAT
        ip_proto = int.from_bytes(payload[0:1], "big")
        size_pkts = list()
        idx = 1
        for i in range(self.FIRST_N_PKTS):
            last_idx = idx + self.BYTES_PKT_SIZE
            size_pkt = int.from_bytes(payload[idx:last_idx], "big")
            size_pkts.append(size_pkt)
            idx = last_idx
        iat_pkts = list()
        for i in range(self.FIRST_N_PKTS-1):
            last_idx = idx + self.BYTES_PKT_IAT
            iat_pkt = int.from_bytes(payload[idx:last_idx], "big")
            iat_pkts.append(iat_pkt)
            idx = last_idx

        # If no TCP nor UDP, update source and destinations ports
        if ip_proto != in_proto.IPPROTO_TCP or ip_proto != in_proto.IPPROTO_UDP:
            port_src = self.PORT_MISSED_TCP_UDP
            port_dst = self.PORT_MISSED_TCP_UDP

        self.logger.info(
            "  Reported elephant flow: IP_src:%s, IP_dst:%s, IP_protocol:%d, "
            + "Port_src:%d, Port_dst:%d, Size_pkts:%s, IAT_pkts:%s",
            ip_src, ip_dst, ip_proto, port_src, port_dst, size_pkts, iat_pkts
        )

        # Ignore report if flowID already exists in flow table
        flowid = self.build_flowid(
            ip_src, ip_dst, ip_proto, port_src, port_dst
        )
        self.lock_eflows.acquire()
        if flowid in self.eflows:
            self.logger.info(
                "  Ignoring as elephant flow (%s) alredy exists in flow table",
                flowid
            )
            self.lock_eflows.release()
            return
        self.lock_eflows.release()

        # Estimate/predict rate and duration of the elephant flow
        x, y_hat_rate, y_hat_time = self.estimate_predict_eflow(
            ip_src, ip_dst, ip_proto, port_src, port_dst, size_pkts, iat_pkts
        )

        self.logger.info(
            "  Estimated/predicted values: Rate=%f %s, Duration=%f seconds",
            y_hat_rate, self.UNIT_RATE, y_hat_time
        )

        # Select path (using predicted data) and check it goes through an
        # Aggregate switch, at least
        path = self.select_path_eflow(dpid_src, dpid_dst, y_hat_rate)
        if path is None or len(path) < self.FATTREE_LEN_PATH_POD_EDGE:
            self.logger.warning(
                " No paths available from Switch (%s) to Switch (%s)",
                self.get_dpid_string(dpid_src), self.get_dpid_string(dpid_dst)
            )
            return

        # Build match arguments
        match_args = self.build_match_eflow(
            ip_src, ip_dst, ip_proto, port_src, port_dst
        )

        # Install flow rule in entry Edge switch (with flow removed event)
        dpid_edge = path[0]
        dpid_aggr = path[1]
        port_etoa = self.td_network[dpid_edge][dpid_aggr]["port"]
        self.add_rule_eflow(dp, match_args, port_etoa, 1, True)

        # Check path goes through a Core switch (no flow removed event)
        if len(path) > self.FATTREE_LEN_PATH_POD_EDGE:
            # Install flow rule in selected Aggregate switch
            dp_aggr = self.switch_info[dpid_aggr]["dp"]
            dpid_core = path[2]
            port_atoc = self.td_network[dpid_aggr][dpid_core]["port"]
            self.add_rule_eflow(dp_aggr, match_args, port_atoc)

        # Add elephant flow to list with data
        end_time = time.time() + y_hat_time
        self.eflows[flowid] = {
            "x": x,
            "rate": y_hat_rate,
            "end_time": end_time,
            "path": path
        }

        # Add flow to selected path (in both directions)
        self.lock_eflows.acquire()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            self.td_network[u][v]["eflows"].append(flowid)
            self.td_network[v][u]["eflows"].append(flowid)
        self.lock_eflows.release()

    # -------------------------------------------------------------------------
    # Estimate or predict elephant flow rate and prediction
    # -------------------------------------------------------------------------
    def estimate_predict_eflow(self, ip_src, ip_dst, ip_proto, port_src,
                               port_dst, size_pkts, iat_pkts):
        # Parse input data
        x = self.feats_to_input_data(ip_src, ip_dst, ip_proto, port_src,
                                     port_dst, size_pkts, iat_pkts)

        # Estimate rate from first packets during cold start
        if self.train_count < self.COLD_START_THRESHOLD:
            self.logger.info(
                "  Cold start: remain %d trains for using ML prediction",
                self.COLD_START_THRESHOLD - self.train_count
            )
            # Estimate using size and iat of first N packets
            bitcount_npkts = sum(size_pkts) * self.BITS_IN_BYTE  # bits
            duration_npkts = sum(iat_pkts) / self.USEC_IN_SEC  # seconds
            y_hat_rate = (
                    (bitcount_npkts / self.EFLOW_RATE_DIVISOR) / duration_npkts
            )
            if y_hat_rate > self.MAX_RATE:
                y_hat_rate = self.MAX_RATE
            y_hat_time = self.DEFAULT_DURATION
            return x, y_hat_rate, y_hat_time

        # Predict rate and duration using input data (lock models)
        self.lock_models.acquire()
        y_hat_rate = self.model_rate.predict_on_batch(x)
        y_hat_time = self.model_time.predict_on_batch(x)
        self.lock_models.release()

        # Clip prediction to a minimum value
        y_hat_rate = np.clip(
            y_hat_rate, a_min=self.MIN_RATE, a_max=self.MAX_RATE
        )
        y_hat_time = np.clip(
            y_hat_time, a_min=self.MIN_DURATION, a_max=self.MAX_DURATION
        )

        return x, y_hat_rate.flatten()[0], y_hat_time.flatten()[0]

    # -------------------------------------------------------------------------
    # Select the best path between the Edge switches either by computing the
    # least congested path or by using the rate to select the best fit
    # -------------------------------------------------------------------------
    def select_path_eflow(self, edge_src, edge_dst, rate):
        # Check one or more paths exist between Edge switches
        if len(self.edge_paths[(edge_src, edge_dst)]) <= 0:
            return None

        # Find least-congested path between Edge switches
        # return self.least_congested_path(edge_src, edge_dst)

        # Find best path between Edge switches that fits the rate
        return self.worst_fit_bwtime_fit_path(edge_src, edge_dst, rate)

    # -------------------------------------------------------------------------
    # Least-congested path between Edge switches
    # -------------------------------------------------------------------------
    def least_congested_path(self, edge_src, edge_dst):
        minload_path = self.edge_paths[(edge_src, edge_dst)][0]
        min_load = sys.maxsize
        self.lock_eflows.acquire()
        for path in self.edge_paths[(edge_src, edge_dst)]:
            # Compute path load
            _, path_load = self.get_path_load(path)

            # Check if a lesser congested path was found
            if path_load < min_load:
                minload_path = path
                min_load = path_load
        self.lock_eflows.release()
        return minload_path

    # -------------------------------------------------------------------------
    # Worst-Fit Badwidth-Time-Fit (WF-BTF) scheduling algorithm. It selects the
    # path with the minimum load (i.e., least-congested path) if it can fit the
    # requested rate. If the path cannot fit the rate, it selects the path with
    # the maximum harmonic mean between the relative scores computed for the
    # free bandwidth and for the remaining time to fit the requested rate. Each
    # relative score is computed using the best value as a reference: maximum
    # free bandwidth and minimum remaining time
    # -------------------------------------------------------------------------
    def worst_fit_bwtime_fit_path(self, edge_src, edge_dst, rate):
        # Find least congested path
        maxbw_path = self.Path(self.edge_paths[(edge_src, edge_dst)][0])
        max_freebw = 0
        paths = list()
        self.lock_eflows.acquire()
        for p in self.edge_paths[(edge_src, edge_dst)]:
            # Get elephant flows traversing the path and compute path load
            path_eflows, path_load = self.get_path_load(p)

            # Check if a lesser congested path was found
            path = self.Path(p, path_eflows, path_load)
            paths.append(path)
            if path.free_bw > max_freebw:
                maxbw_path = path
                max_freebw = path.free_bw
        self.lock_eflows.release()

        # Check if rate fits in least congested path
        if rate <= maxbw_path.free_bw:
            return maxbw_path.path

        # Compute time to fit rate
        min_fittime = sys.maxsize
        now_time = time.time()
        self.lock_eflows.acquire()
        for path in paths:
            fit_time = path.compute_fit_time(now_time, rate)
            if fit_time < min_fittime:
                min_fittime = fit_time
        self.lock_eflows.release()

        # Find path with the highest bandwidth-duration harmonic mean
        maxhmean_path = paths[0]
        max_hmean = 0
        for path in paths:
            hmean = path.compute_harmonic_mean(max_freebw, min_fittime)
            if hmean > max_hmean:
                maxhmean_path = path
                max_hmean = hmean
        return maxhmean_path.path

    # -------------------------------------------------------------------------
    # Find elephant flows traversing the path and compute path load
    # -------------------------------------------------------------------------
    def get_path_load(self, path):
        # Find (unique) elephant flows traversing the path
        eflow_ids = set()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for flowid in self.td_network[u][v]["eflows"]:
                eflow_ids.add(flowid)

        # Compute load from elephant flows in the path
        eflows = list()
        load = 0
        for flowid in eflow_ids:
            eflows.append(self.eflows[flowid])
            load += self.eflows[flowid]["rate"]
        return eflows, load

    # -------------------------------------------------------------------------
    # Path object that stores the sequence of switches in the path, the set of
    # elephant flows traversing the path, the available bandwidth, and the
    # remaining time to fit the requested rate
    # -------------------------------------------------------------------------
    class Path:
        LINK_BW = 1_000  # Kbps TODO set to IRide.MAX_RATE

        def __init__(self, path, eflows=(), load=LINK_BW):
            self.path = path
            self.eflows = eflows
            self.free_bw = self.adjust_value(self.LINK_BW - load)
            self.fit_time = sys.maxsize

        def compute_fit_time(self, now, rate):
            eflows_bytime = sorted(self.eflows, key=lambda e: e["end_time"])
            freed_bw = self.free_bw
            for eflow in eflows_bytime:
                freed_bw += eflow["rate"]
                if rate <= freed_bw:
                    self.fit_time = self.adjust_value(eflow["end_time"] - now)
                    break
            return self.fit_time

        def compute_harmonic_mean(self, max_freebw, min_fittime):
            freebw_score = self.free_bw / max_freebw
            fittime_score = min_fittime / self.fit_time
            return (2 * (freebw_score * fittime_score)
                    / (freebw_score + fittime_score))

        @staticmethod
        def adjust_value(value):
            if value == 0:
                return 1
            elif value < 0:
                return 1 / abs(value)
            return value

    # -------------------------------------------------------------------------
    # Add flow rule for reported elephant flow
    # -------------------------------------------------------------------------
    def add_rule_eflow(self, dp, match_args, out_port, in_table=0,
                       remove_event=False):
        # Build actions and check if should generate flow removal event
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch(**match_args)
        actions = [ofp_parser.OFPActionOutput(out_port)]
        flags = 0
        if remove_event:
            ofp = dp.ofproto
            flags = ofp.OFPFF_SEND_FLOW_REM

        # Add flow rule in table 1 with the corresponding idle timeout
        self.add_flow(
            dp, self.PRIOR_EFLOW, match, actions, in_table=in_table,
            idle_to=self.IDLETO_EFLOW, flags=flags, cookie=self.COOKIE_EFLOW
        )

        self.logger.info(
            "  Swicth (%s) flow rule: Table:%d Priority:%d IdleTO:%d Flags:%s "
            + "Cookie:%d [EFLOW:%s] -> Port:%s",
            self.get_dpid_string(dp.id), in_table, self.PRIOR_EFLOW,
            self.IDLETO_EFLOW, flags, self.COOKIE_EFLOW, match_args, out_port
        )

    # -------------------------------------------------------------------------
    # RYU event: flow removed handler. Use the information from timed-out rules
    # for elephant flows to build the ground truth and incrementally train the
    # machine learning models for rate and duration
    # -------------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        # Check event is due to idle timeout
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        if msg.reason != ofp.OFPRR_IDLE_TIMEOUT:
            return

        # Check event was generated by a timed-out elephant flow rule
        if msg.cookie != self.COOKIE_EFLOW:
            return

        # Get flow ID
        flowid = self.get_flowid_from_match(msg.match)
        self.logger.info(
            "\nReceived timed-out flow (%s) in Switch (%s)",
            flowid, self.get_dpid_string(dp.id)
        )

        # Check timed-out flow is an elephant
        x = self.eflows[flowid]["x"]
        flow_size = x[0][104:111].sum() + msg.byte_count
        if flow_size < self.EFLOW_SIZE_THRESHOLD:
            self.logger.info(
                "  Not an elephant flow, flow size = %s bytes", flow_size
            )
            # Remove flow from path and table
            self.remove_eflow(flowid)
            return

        # Get rate and duration
        rate, duration = self.get_rate_duration(msg, self.IDLETO_EFLOW)
        self.logger.info(
            "  Train models using elephant flow ground truth: "
            + "Rate=%f %s, Duration=%f seconds",
            rate, self.UNIT_RATE, duration
        )

        # Start thread for training the models
        train = threading.Thread(
            target=self.train_models, args=(flowid, rate, duration)
        )
        train.start()

    # -------------------------------------------------------------------------
    # Train the neural network models for rate and duration using the input
    # data of a flow ID and the ground truth of rate and duration
    # -------------------------------------------------------------------------
    def train_models(self, flowid, rate, duration):
        # Build training data
        x = self.eflows[flowid]["x"]
        y_rate = np.array([[rate]])
        y_time = np.array([[duration]])
        self.x_train = tf.concat([self.x_train, x], 0)
        self.y_train_rate = tf.concat([self.y_train_rate, y_rate], 0)
        self.y_train_time = tf.concat([self.y_train_time, y_time], 0)
        while self.x_train.shape[0] > self.BATCH_SIZE:
            self.x_train = self.x_train[1:]
            self.y_train_rate = self.y_train_rate[1:]
            self.y_train_time = self.y_train_time[1:]

        # Train the models (using the copy)
        self.cpmodel_rate.train_on_batch(self.x_train, self.y_train_rate)
        self.cpmodel_time.train_on_batch(self.x_train, self.y_train_time)

        # Copy weights from trained models (lock when copying)
        self.lock_models.acquire()
        self.model_rate.set_weights(self.cpmodel_rate.get_weights())
        self.model_time.set_weights(self.cpmodel_time.get_weights())
        self.lock_models.release()

        # Increment training counter and remove flow from path and table
        self.train_count += 1
        self.remove_eflow(flowid)

    # -------------------------------------------------------------------------
    # Remove elephant flow from assigned path and from table of elephant flows
    # -------------------------------------------------------------------------
    def remove_eflow(self, flowid):
        path = self.eflows[flowid]["path"]
        self.lock_eflows.acquire()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            self.td_network[u][v]["eflows"].remove(flowid)
            self.td_network[v][u]["eflows"].remove(flowid)
        del self.eflows[flowid]
        self.lock_eflows.release()
        self.logger.info("  Removed flow from both path and flow table")

    # -------------------------------------------------------------------------
    # Overriden: perform PM2 LLDP handling and start elephant flow monitoring
    # threads on Edge switches, if the layer definition has been completed
    # -------------------------------------------------------------------------
    def lldp_packet_handler(self, msg, pkt):
        # Perform PM2 LLDP packets handling
        super(IRide, self).lldp_packet_handler(msg, pkt)

        # Check if layer definition has not been completed
        if not self.layerdef_complete:
            return

        # Build network paths between Edge switches
        self.logger.info(
            "\nConfiguring network paths between pairs of Edge switches"
        )
        for edge_src in self.edges:
            for edge_dst in self.edges:
                # Check destination edge is not the same source edge
                if edge_src.id != edge_dst.id:
                    # Recursively find paths between a pair of edge switches
                    self.edge_paths.setdefault((edge_src.id, edge_dst.id), [])
                    self.find_edge_paths(
                        edge_src.id, edge_dst.id, edge_src.id, "", tuple()
                    )
                    self.logger.info(
                        "  From Edge switch (%s) to Edge switch (%s)",
                        self.get_dpid_string(edge_src.id),
                        self.get_dpid_string(edge_dst.id)
                    )

        # Initialize list of elephant flows assigned to links between switches
        for u, v in self.td_network.edges():
            self.td_network[u][v]["eflows"] = list()
        self.logger.info(
            "Completed existing network paths between Edge switches"
        )

        # Check if monitoring rate has been established
        if self.MONITOR_RATE_EFLOW <= 0:
            self.logger.info(
                "\nNo elephant flow monitoring on Edge switches"
            )
            return

        # Start an elephant flow monitoring thread per Edge switch
        self.logger.info(
            "\nStarting elephant flow monitoring on Edge switches"
        )
        for edge in self.edges:
            monitor = threading.Thread(
                target=self.edge_eflows_monitor, args=(edge,)
            )
            monitor.start()
            self.logger.info(
                "  Started on Edge switch %s", self.get_dpid_string(edge.id)
            )

    # -------------------------------------------------------------------------
    # Recursive function to find networks paths between a pair of Edge switches
    # -------------------------------------------------------------------------
    def find_edge_paths(self, edge_src, edge_dst, dp_now, dp_prev, path_prev):
        for u, v in self.td_network.edges():
            if u == dp_now and v != dp_prev:
                path_now = path_prev + (u,)
                if v == edge_dst:
                    path = path_now + (v,)
                    self.edge_paths[(edge_src, edge_dst)].append(path)
                elif len(path_now) < self.FATTREE_LEN_PATH_EDGE:
                    self.find_edge_paths(edge_src, edge_dst, v, u, path_now)

    # -------------------------------------------------------------------------
    # Elephant flows monitoring thread per Edge switch
    # -------------------------------------------------------------------------
    def edge_eflows_monitor(self, dp):
        # Build flow stats request for elephant flow rules
        in_table = 1
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP)
        request = ofp_parser.OFPFlowStatsRequest(
            datapath=dp, table_id=in_table, out_port=ofp.OFPP_ANY,
            cookie=self.COOKIE_EFLOW, match=match
        )

        # While the switch is connected, send request at the monitoring rate
        while True:
            if dp.id not in self.switch_info:
                break
            dp.send_msg(request)
            time.sleep(self.MONITOR_RATE_EFLOW)

    # -------------------------------------------------------------------------
    # RYU event: flow stats reply handler
    # -------------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        for msg in ev.msg.body:
            # Get flow ID and rate to update
            flowid = self.get_flowid_from_match(msg.match)
            rate, _ = self.get_rate_duration(msg)
            self.lock_eflows.acquire()
            self.eflows[flowid]["rate"] = rate
            self.lock_eflows.release()
            self.logger.info(
                "\nReceived stats from flow (%s). Updating Rate=%f %s",
                flowid, rate, self.UNIT_RATE
            )

    # -------------------------------------------------------------------------
    # Build neural network model (with a copy) using tensorflow and the passed
    # arguments: number of input units, number of hidden layers, number of
    # hidden units per each hidden layer, regularization parameter lambda, and
    # dropout rate per each hidden layer
    # -------------------------------------------------------------------------
    @staticmethod
    def build_tensorflow_model(input_units, hidden_layers, hidden_units,
                               lmbda=0.0, dropout=0.0):
        # Initialize model
        model = tf.keras.models.Sequential()
        # Add hidden layers
        for i in range(hidden_layers):
            # Check if first hidden layer
            if i == 0:
                input_shape = (input_units,)
            else:
                input_shape = None
            model.add(
                IRide.build_dense_layer(input_shape, hidden_units, lmbda)
            )
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))

        # Add output layer
        model.add(tf.keras.layers.Dense(1))

        # Make a copy of the model (used for training)
        copy_model = tf.keras.models.clone_model(model)

        # Compile model (and copy)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(),
                      metrics=["mse"])
        copy_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(),
                           metrics=["mse"])
        return model, copy_model

    # -------------------------------------------------------------------------
    # Build a dense layer using the passed arguments: input units, number of
    # hidden units, regularization parameter lambda, unit activation function,
    # and unit initializer
    # -------------------------------------------------------------------------
    @staticmethod
    def build_dense_layer(input_shape, hidden_units, lmbda, activation="relu",
                          initializer="he_normal"):
        # Create layer without or with input shape
        if input_shape is None:
            layer = tf.keras.layers.Dense(hidden_units,
                                          activation=activation,
                                          kernel_initializer=initializer)
        else:
            layer = tf.keras.layers.Dense(hidden_units,
                                          activation=activation,
                                          kernel_initializer=initializer,
                                          input_shape=input_shape)

        # Add regularization to layer if requested
        if lmbda > 0:
            layer.kernel_regularizer = tf.keras.regularizers.l2(lmbda)
        return layer

    # -------------------------------------------------------------------------
    # Parse the passed features to neural network input data
    # -------------------------------------------------------------------------
    @staticmethod
    def feats_to_input_data(ip_src, ip_dst, ip_proto, port_src, port_dst,
                            pkt_sizes, pkt_iats):
        # Parse 5-tuple header (replace 0s by -1s)
        binheader = IRide.ip_to_bits(ip_src)
        binheader.extend(IRide.ip_to_bits(ip_dst))
        binheader.extend(IRide.to_bits("{:08b}", ip_proto))
        binheader.extend(IRide.to_bits("{:016b}", port_src))
        binheader.extend(IRide.to_bits("{:016b}", port_dst))

        x = np.array(binheader)
        x = np.where(x == 0, -1, x)

        # Append size and IAT of first N packets
        x = np.append(x, pkt_sizes)
        x = np.append(x, pkt_iats)

        # Resize array
        return np.expand_dims(x, axis=0)

    # -------------------------------------------------------------------------
    # Parse an IP to bit representation
    # -------------------------------------------------------------------------
    @staticmethod
    def ip_to_bits(ip):
        bits = list()
        octets = ip.split(".")
        for octet in octets:
            bits.extend(IRide.to_bits("{:08b}", octet))
        return bits

    # -------------------------------------------------------------------------
    # Parse a value to bit representation using a string format
    # -------------------------------------------------------------------------
    @staticmethod
    def to_bits(string_format, value):
        binary = string_format.format(int(value))
        return [int(b) for b in binary]

    # -------------------------------------------------------------------------
    # Build match arguments for reported elephant flow
    # -------------------------------------------------------------------------
    @staticmethod
    def build_match_eflow(ip_src, ip_dst, ip_proto, port_src, port_dst):
        match = {
            "eth_type": ether_types.ETH_TYPE_IP,
            "ipv4_src": ip_src,
            "ipv4_dst": ip_dst,
            "ip_proto": ip_proto
        }
        if ip_proto == in_proto.IPPROTO_TCP:
            match["tcp_src"] = port_src
            match["tcp_dst"] = port_dst
        elif ip_proto == in_proto.IPPROTO_UDP:
            match["udp_src"] = port_src
            match["udp_dst"] = port_dst
        return match

    # -------------------------------------------------------------------------
    # Build flow ID for reported elephant flow
    # -------------------------------------------------------------------------
    @staticmethod
    def build_flowid(ip_src, ip_dst, ip_proto, port_src, port_dst):
        return "{ip_src}_{ip_dst}|{ip_proto}|{port_src}_{port_dst}".format(
            ip_src=ip_src, ip_dst=ip_dst, ip_proto=ip_proto, port_src=port_src,
            port_dst=port_dst
        )

    # -------------------------------------------------------------------------
    # Get flow ID from the fields in a match object
    # -------------------------------------------------------------------------
    @staticmethod
    def get_flowid_from_match(match):
        ip_src = ip_dst = ""
        ip_proto = port_src = port_dst = 0
        for key, value in match._fields2:
            if key == "ipv4_src":
                ip_src = value
            elif key == "ipv4_dst":
                ip_dst = value
            elif key == "ip_proto":
                ip_proto = value
            elif key == "tcp_src":
                port_src = value
            elif key == "udp_src":
                port_src = value
            elif key == "tcp_dst":
                port_dst = value
            elif key == "udp_dst":
                port_dst = value
        return IRide.build_flowid(ip_src, ip_dst, ip_proto, port_src, port_dst)

    # -------------------------------------------------------------------------
    # Get rate (in units specified by the divisor) and duration (in seconds)
    # from an openflow message
    # -------------------------------------------------------------------------
    @staticmethod
    def get_rate_duration(msg, idle_to=0):
        duration = msg.duration_sec - idle_to  # seconds
        if duration <= 0:
            duration = 1
        bit_count = msg.byte_count * IRide.BITS_IN_BYTE  # bits
        rate = (bit_count / IRide.EFLOW_RATE_DIVISOR) / duration
        return rate, duration
