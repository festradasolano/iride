# =============================================================================
#
#    Pseudo-MAC Multipath (PM2) Routing
#
# =============================================================================

__title__ = "Pseudo-MAC Multipath (PM2) Routing"
__author__ = "cfamezquita,festradasolano"
__year__ = "2013-2021"

import math
import sys
import time
import networkx as nx

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, lldp, arp
from ryu.lib.packet import ether_types

# ==============================
# Main API Class (OpenFlow v1.3)
# ==============================

_threshold = 60


class PM2Routing(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    LLDP_TIMEOUT = 4

    LVL_UNKN = 0
    LVL_CORE = 1
    LVL_AGGR = 2
    LVL_EDGE = 3

    LEVELS = {
        LVL_CORE: "CORE",
        LVL_AGGR: "AGGREGATE",
        LVL_EDGE: "EDGE",
        LVL_UNKN: "UNKNOWN"
    }

    PRIOR_ARP = 65535
    PRIOR_PMAC = 300
    PRIOR_PMACMISS = 100
    PRIOR_TABLEMISS = 0

    INIT_POD = 0
    INIT_POS = 0
    INIT_VMID = 0

    ARP_OPER_REQUEST = 1
    ARP_OPER_REPLY = 2

    # -------------------------------------------------------------------------
    # Initialization Process
    # -------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super(PM2Routing, self).__init__(*args, **kwargs)

        self.sw_count = 0
        self.fattree_k = 0

        self.td_network = nx.DiGraph()
        self.td_edgepos = {}

        self.cores = []
        self.aggregates = []
        self.edges = []

        self.ports_e2a = {}
        self.ports_a2e = {}
        self.ports_a2c = {}
        self.ports_c2a = {}
        self.ports_to_pod = {}
        self.ports_to_pos = {}

        self.ip_table = {}
        self.vm_table = {}
        self.pending_reqs = {}

        self.installed_rules = []

        self.switch_info = {}
        self.init_time = 0
        self.time_table = {}

        # LLDP variables
        self.lldp_timeout = self.LLDP_TIMEOUT
        self.lldp_neighbors_threshold = sys.maxsize
        self.lldp_neighbors_count = 0
        self.lldp_neighbors = {}
        self.lldp_level = {}

        self.lldp_complete = False
        self.layerdef_complete = False

    # -------------------------------------------------------------------------
    # RYU event: switch features
    # -------------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        # Add switch to wait list of connections
        dpid = ev.msg.datapath.id
        self.sw_count += 1
        self.logger.info(
            "Switch %s added to waitlist. Waiting %d switches to connect",
            self.get_dpid_string(dpid), self.sw_count
        )

    # -------------------------------------------------------------------------
    # RYU event: switch enter
    # -------------------------------------------------------------------------
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        # Switch info
        switch = ev.switch
        num_ports = len(switch.ports)
        self.sw_count -= 1
        self.logger.info(
            "\nSwitch %s connected, Ports: %d. Waiting %d switches to connect",
            self.get_dpid_string(switch.dp.id), num_ports, self.sw_count
        )

        # Use number of ports in a switch to identify fat-tree topology size,
        # including number of neighbors for LLDP to complete
        if self.fattree_k == 0 and num_ports > self.fattree_k:
            self.fattree_k = num_ports
            n_cores = int((self.fattree_k//2)**2)
            n_aggrs = n_edges = int((self.fattree_k//2) * self.fattree_k)
            self.lldp_neighbors_threshold = int(
                    (n_cores * self.fattree_k)
                    + (n_aggrs * self.fattree_k)
                    + (n_edges * self.fattree_k/2)
            )
            self.logger.info(
                "  Identified size of fat-tree topology K = %d. Switches per "
                + "layer: Core=%d, Aggregate=%d, Edge=%d",
                self.fattree_k, n_cores, n_aggrs, n_edges
            )
        # If different number of ports, wait enough time for LLDP to complete
        elif num_ports != self.fattree_k:
            if num_ports > self.fattree_k:
                self.fattree_k = num_ports
            self.lldp_neighbors_threshold = 0
            self.lldp_timeout = (
                    (self.LLDP_TIMEOUT ** math.log2(self.fattree_k))
                    + (self.LLDP_TIMEOUT * self.fattree_k)
            )
            self.logger.info(
                "  Identified non-homogeneous switch port number in fat-tree "
                + "topology. Going to wait %d seconds for LLDP to complete",
                self.lldp_timeout
            )

        # Initialize LLDP attributes for switch
        self.lldp_neighbors.setdefault(switch.dp.id, [])
        self.lldp_level.setdefault(switch.dp.id, self.LVL_UNKN)
        self.switch_info[switch.dp.id] = {
            "dp": switch.dp,
            "ports": num_ports
        }

        # Check if no more switches in wait list
        if self.sw_count == 0:
            msg = "Waiting for {th} ports to receive LLDP packets".format(
                th=self.lldp_neighbors_threshold
            )
            if self.lldp_neighbors_threshold == 0:
                msg = "Waiting for {to} seconds to complete LLDP".format(
                    to=self.lldp_timeout
                )
            self.init_time = time.time()
            self.logger.info(
                "  All switches in waitlist have been connected. %s", msg
            )

    # -------------------------------------------------------------------------
    # RYU event: switch leave
    # -------------------------------------------------------------------------
    @set_ev_cls(event.EventSwitchLeave, [
        MAIN_DISPATCHER, CONFIG_DISPATCHER, DEAD_DISPATCHER
    ])
    def switch_leave_handler(self, ev):
        switch = ev.switch
        del self.switch_info[switch.dp.id]
        self.logger.info(
            "\nSwitch %s disconnected. Remain %d switches connected",
            self.get_dpid_string(switch.dp.id), len(self.switch_info)
        )

    # -------------------------------------------------------------------------
    # RYU event: packet-in handler. Returns `True` if the LLDP or ARP packets
    # are handled; otherwise, returns `False`
    # -------------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        # Get Message Data
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        # Handle LLDP packet if switch layer definition is not completed
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            if not self.layerdef_complete:
                self.lldp_packet_handler(msg, pkt)
            return True
        # Handle ARP packet
        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            self.arp_packet_handler(msg, pkt)
            return True
        # No more packets to handle
        return False

    # -------------------------------------------------------------------------
    # LLDP packet handler
    # -------------------------------------------------------------------------
    def lldp_packet_handler(self, msg, pkt):
        # Get LLDPDU Information
        #
        # It is obtained as a matrix of objects in tlvs, where
        # position 0 is the Chassis ID object, 1 is the Port ID
        # object, 2 is the TTL object, 3 is the first TLV, 4 is
        # the second TLV and so on. The TLVs are optional and
        # may be inserted in any order. The matrix ends with an
        # End object, which can be obtained by tlvs[-1].
        #
        # To obtain the dpid of the LLDPDU, is needed to get
        # first the information from the Chassis ID. It is a
        # string headed by 'dpid:', so its necessary to replace
        # it with '0x' so it can be parsed easily as a hex value.

        dpid = msg.datapath.id
        pkt_lldp = pkt.get_protocol(lldp.lldp)
        lldp_tlvs = pkt_lldp.tlvs
        lldp_dpid = int(lldp_tlvs[0].chassis_id.replace(b"dpid:", b"0x"), 16)

        # Update neighbor list
        if lldp_dpid not in self.lldp_neighbors[dpid]:
            self.lldp_neighbors_count += 1
            self.lldp_neighbors[dpid].append(lldp_dpid)

        # Check if LLDP has not been completed
        if not self.lldp_complete:
            # Check if all neighbors have been discovered
            if self.lldp_neighbors_count < self.lldp_neighbors_threshold:
                return

            # Check time since no more switches connected
            elapsed_time = time.time() - self.init_time
            if elapsed_time < self.lldp_timeout:
                return

            # Execute first time after LLDP completed
            self.lldp_complete = True
            self.logger.info(
                "\nLLDP completed! Defining switch layers using LLDP data"
            )

        # Check if switch level has been already assigned
        if self.lldp_level[dpid] != self.LVL_UNKN:
            return

        # Check if EDGE switch: switch that did not receive LLDP packets
        # through half or more of the ports
        neighbors = len(self.lldp_neighbors[dpid])
        if neighbors <= self.switch_info[dpid]["ports"] / 2:
            self.set_switch_level(dpid, self.LVL_EDGE, self.edges)

        # Check if either AGGREGATE or CORE: switches that received LLDP
        # packets through all ports
        elif neighbors == self.switch_info[dpid]["ports"]:

            # AGGREGATE switch: LLDP packet received from edge switch
            if self.lldp_level[lldp_dpid] == self.LVL_EDGE:
                self.set_switch_level(dpid, self.LVL_AGGR, self.aggregates)

            # CORE switch: LLDP packet received from aggregate switch
            elif self.lldp_level[lldp_dpid] == self.LVL_AGGR:
                self.set_switch_level(dpid, self.LVL_CORE, self.cores)

        # Check if switch layer definition is complete
        for dpid, level in list(self.lldp_level.items()):
            if level == self.LVL_UNKN:
                return
        self.layerdef_complete = True

        # Build network topology
        self.build_network_graph()
        # self.set_edges_position_auto()
        self.set_edges_position_dpid()  # used for testing

        # Topology configuration
        self.configure_topo_ports()
        self.configure_topo_rules()

        # Check time to complete and restart initial time
        conf_time = time.time() - self.init_time
        self.init_time = time.time()
        self.logger.info(
            "\nTopology configuration completed %f seconds after the last "
            + "switch was connected", conf_time
        )

    # -------------------------------------------------------------------------
    # Set the passed level to the switch and add the switch to the passed list
    # -------------------------------------------------------------------------
    def set_switch_level(self, dpid, lvl, lvl_switches):
        self.lldp_level[dpid] = lvl
        self.logger.info("  Switch (%s) is %s", self.get_dpid_string(dpid),
                         self.LEVELS.get(lvl, "UNDEFINED"))
        lvl_switches.append(self.switch_info[dpid]["dp"])

    # -------------------------------------------------------------------------
    # Build network graph from topology data
    # -------------------------------------------------------------------------
    def build_network_graph(self):
        # We get the list of switch and link objects from the topology
        # API and make lists with the main parameters of each component.
        # Then we use the networkx library to update the network graph
        # with the topology data. It is necessary to run the application
        # with --observe-links for this function to work properly.
        #
        # More info at:
        # https://sdn-lab.com/2014/12/31/topology-discovery-with-ryu/
        #
        self.logger.info("\nUpdating topology discovery network graph")

        # Add switches as nodes
        switch_list = get_switch(self, None)
        topo_switches = [s.dp.id for s in switch_list]
        self.td_network.add_nodes_from(topo_switches)

        # Add links as edges
        links_list = get_link(self, None)
        topo_links1 = [
            (link.src.dpid, link.dst.dpid, {"port": link.src.port_no})
            for link in links_list
        ]
        topo_links2 = [
            (link.dst.dpid, link.src.dpid, {"port": link.dst.port_no})
            for link in links_list
        ]
        self.td_network.add_edges_from(topo_links1)
        self.td_network.add_edges_from(topo_links2)

    # -------------------------------------------------------------------------
    # Set position of Edge switches using DPIDs configured manually. This is
    # useful for testing purposes
    # -------------------------------------------------------------------------
    def set_edges_position_dpid(self):
        self.logger.info("\nSetting position of Edge switches")
        for edge in self.edges:
            s_dpid = self.get_dpid_string(edge.id)
            pod = int(s_dpid[8:12], base=16)
            pos = int(s_dpid[12:], base=16)
            self.td_edgepos[edge.id] = {"pod": pod, "pos": pos}
            self.logger.info("  Edge switch (%s) position: %s",
                             self.get_dpid_string(edge.id),
                             self.td_edgepos[edge.id])

    # -------------------------------------------------------------------------
    # Set position of Edge switches automatically disregarding DPIDs. This is
    # useful for not relying on manual DPID configuration
    # -------------------------------------------------------------------------
    def set_edges_position_auto(self):
        self.logger.info("\nSetting position of Edge switches")
        for edge in self.edges:
            # Check if a switch from the same pod has been assigned with a pod
            # and position; if so, get the pod and max position assigned
            pod, maxpos = self.get_maxpos_in_pod(edge)
            if maxpos is not None:
                self.td_edgepos[edge.id] = {"pod": pod, "pos": maxpos + 1}
            else:
                # Check if a pod has been to assigned; if so, get the max pod
                maxpod = self.get_maxpod()
                if maxpod is not None:
                    self.td_edgepos[edge.id] = {"pod": maxpod + 1,
                                                "pos": self.INIT_POS}
                # First pod and position to assign
                else:
                    self.td_edgepos[edge.id] = {"pod": self.INIT_POD,
                                                "pos": self.INIT_POS}
            self.logger.info("  Edge switch (%s) position: %s",
                             self.get_dpid_string(edge.id),
                             self.td_edgepos[edge.id])

    # -------------------------------------------------------------------------
    # Check maximum position assigned in a Pod
    # -------------------------------------------------------------------------
    def get_maxpos_in_pod(self, dp):
        values = []
        nearest_edges = self.get_edges_nearest(dp.id)
        pod = self.get_pod_from_edges(nearest_edges)
        if pod is not None:
            pod_table = [
                entry for entry in list(self.td_edgepos.values())
                if entry["pod"] == pod
            ]
            for entry2 in pod_table:
                values.append(entry2["pos"])
            return pod, max(values)
        else:
            return None, None
        pass

    # -------------------------------------------------------------------------
    # Get nearest Edge switches
    # -------------------------------------------------------------------------
    def get_edges_nearest(self, dpid):
        # First Hops
        first_hops = [hop for hop in self.td_network.edges() if hop[0] == dpid]

        # Second Hops
        edges = []
        for h in first_hops:
            edges += [hop[1] for hop in self.td_network.edges()
                      if hop[0] == h[1]
                      and self.lldp_level[hop[1]] == self.LVL_EDGE
                      and hop[1] != dpid]
        return list(set(edges))

    # -------------------------------------------------------------------------
    # Get Pod assigned to a set of Edge switches
    # -------------------------------------------------------------------------
    def get_pod_from_edges(self, edges):
        for dpid in edges:
            if dpid in self.td_edgepos:
                return self.td_edgepos[dpid]["pod"]
        return None

    # -------------------------------------------------------------------------
    # Check maximum Pod assigned
    # -------------------------------------------------------------------------
    def get_maxpod(self):
        values = []
        for entry in list(self.td_edgepos.values()):
            values.append(entry["pod"])
        return max(values) if values else None

    # -------------------------------------------------------------------------
    # Identify switch ports connecting switches from Edge to Aggregate, from
    # Aggregate to Edge, from Aggregate to Core, and from Core to Aggregate
    # -------------------------------------------------------------------------
    def configure_topo_ports(self):
        self.logger.info("\nConfiguring topology ports")

        # Edge switches
        for edge in self.edges:
            links = [link for link in self.td_network.edges()
                     if link[0] == edge.id]
            for link in links:
                src = link[0]
                dst = link[1]
                if self.lldp_level[dst] == self.LVL_AGGR:
                    # Get Port of Link from source to destination
                    self.ports_e2a.setdefault(edge.id, [])
                    self.ports_e2a[edge.id].append(
                        self.td_network[src][dst]["port"]
                    )
        self.logger.info("  From EDGE to AGGREGATE")
        self.logger.debug("  %s\n", self.ports_e2a)

        # Aggregate switches
        aggr_pod = {}
        for aggr in self.aggregates:
            ports_pos = {}
            links = [link for link in self.td_network.edges()
                     if link[0] == aggr.id]
            for link in links:
                src = link[0]
                dst = link[1]
                if self.lldp_level[dst] == self.LVL_CORE:
                    # Get Port of Link from source to destination
                    self.ports_a2c.setdefault(aggr.id, [])
                    self.ports_a2c[aggr.id].append(
                        self.td_network[src][dst]["port"]
                    )
                elif self.lldp_level[dst] == self.LVL_EDGE:
                    # Get Port of Link from source to destination
                    self.ports_a2e.setdefault(aggr.id, [])
                    self.ports_a2e[aggr.id].append(
                        (self.td_network[src][dst]["port"], dst)
                    )

                    # Get Pod from Edges
                    if aggr.id not in aggr_pod:
                        aggr_pod[aggr.id] = self.td_edgepos[dst]["pod"]

                    # Get Ports to Edge Positions
                    ports_pos.setdefault(self.td_edgepos[dst]["pos"], [])
                    ports_pos[self.td_edgepos[dst]["pos"]].append(
                        self.td_network[src][dst]["port"])

            self.ports_to_pos[aggr.id] = ports_pos

        self.logger.info("  From AGGREGATE to CORE")
        self.logger.debug("  %s\n", self.ports_a2c)
        self.logger.info("  From AGGREGATE to EDGE")
        self.logger.debug("  %s\n", self.ports_a2e)

        # Core switches
        for core in self.cores:
            self.ports_to_pod.setdefault(core.id, {})
            links = [link for link in self.td_network.edges()
                     if link[0] == core.id]
            for link in links:
                src = link[0]
                dst = link[1]
                if self.lldp_level[dst] == self.LVL_AGGR:
                    # Get Port of Link from source to destination
                    self.ports_c2a.setdefault(core.id, [])
                    self.ports_c2a[core.id].append(
                        (self.td_network[src][dst]["port"], dst)
                    )

                    # Get Ports to Pods
                    self.ports_to_pod[core.id][aggr_pod[dst]] = \
                        self.td_network[src][dst]["port"]

        self.logger.info("  From CORE to AGGREGATE")
        self.logger.debug("  %s\n", self.ports_c2a)
        self.logger.info("  From CORE to POD")
        self.logger.debug("  %s\n", self.ports_to_pod)

    # -------------------------------------------------------------------------
    # Configure topology rules
    # -------------------------------------------------------------------------
    def configure_topo_rules(self):
        # Edge Switches
        self.logger.info("\nConfiguring rules on EDGE switches")
        for edge in self.edges:
            self.add_edge_rules(edge)

        # Aggregate Switches
        self.logger.info("\nConfiguring rules on AGGREGATE switches")
        for aggr in self.aggregates:
            self.add_aggr_rules(aggr)

        # Core Switches
        self.logger.info("\nConfiguring rules on CORE switches")
        for core in self.cores:
            self.add_core_rules(core)

    # -------------------------------------------------------------------------
    # Add rules to Edge switch
    # -------------------------------------------------------------------------
    def add_edge_rules(self, dp):
        # Add group rule: SELECT on ports to Aggregate switches
        self.add_rule_group(dp, 1, self.ports_e2a[dp.id])
        # Add flow rule: ARP -> controller
        self.add_rule_arp(dp)
        # Add flow rule: table 0 miss -> drop
        self.add_rule_tablemiss(dp)
        # Add flow rule: table 1 miss -> drop
        self.add_rule_tablemiss(dp, 1)

        # Add PMAC-miss flow rules
        ofp_parser = dp.ofproto_parser
        pmac, mask = self.generate_pmac_wildcard()
        prior_pmacmiss = self.PRIOR_PMACMISS
        # Ethernet source PMAC-miss table 0 -> table 1
        to_table = 1
        match = ofp_parser.OFPMatch(eth_src=(pmac, mask))
        self.add_flow_table(dp, prior_pmacmiss, to_table, match)
        self.logger.info(
            "  Swicth (%s) flow rule: Table:0 Priority:%d "
            + "[ETH_SRC(%s, %s)] -> Table:%s", self.get_dpid_string(dp.id),
            prior_pmacmiss, pmac, mask, to_table
        )
        # Ethernet destination PMAC-miss table 1 -> group 1
        in_table = 1
        to_group = 1
        match = ofp_parser.OFPMatch(eth_dst=(pmac, mask))
        actions = [ofp_parser.OFPActionGroup(to_group)]
        self.add_flow(dp, prior_pmacmiss, match, actions, in_table)
        self.logger.info(
            "  Swicth (%s) flow rule: Table:%d Priority:%d "
            + "[ETH_DST(%s, %s)] -> Group:%s", self.get_dpid_string(dp.id),
            in_table, prior_pmacmiss, pmac, mask, to_group
        )

    # -------------------------------------------------------------------------
    # Add rules to Aggregate switch
    # -------------------------------------------------------------------------
    def add_aggr_rules(self, dp):
        # Add group rule: SELECT on ports to Core switches
        self.add_rule_group(dp, 1, self.ports_a2c[dp.id])
        # Add flow rule: ARP -> drop
        self.add_rule_arp(dp)
        # Add flow rule: table 0 miss -> drop
        self.add_rule_tablemiss(dp)

        # Add flow rule: PMAC-miss table 0 -> group 1
        ofp_parser = dp.ofproto_parser
        pmac, mask = self.generate_pmac_wildcard()
        prior_pmacmiss = self.PRIOR_PMACMISS
        to_group = 1
        match = ofp_parser.OFPMatch(eth_dst=(pmac, mask))
        actions = [ofp_parser.OFPActionGroup(to_group)]
        self.add_flow(dp, prior_pmacmiss, match, actions)
        self.logger.info(
            "  Swicth (%s) flow rule: Table:0 Priority:%d "
            + "[ETH_DST(%s, %s)] -> Group:%s", self.get_dpid_string(dp.id),
            prior_pmacmiss, pmac, mask, to_group
        )

        # Add flow rules to Edge switches
        ofp_parser = dp.ofproto_parser
        prior_pmac = self.PRIOR_PMAC
        for port in self.ports_a2e[dp.id]:
            # Get Edge switch PMAC wildcard
            pod = self.td_edgepos[port[1]]["pod"]
            pos = self.td_edgepos[port[1]]["pos"]
            pmac, mask = self.generate_pmac_wildcard(pod, pos)

            # Ethernet destination PMAC wildcard -> port to Edge switch
            match = ofp_parser.OFPMatch(eth_dst=(pmac, mask))
            actions = [ofp_parser.OFPActionOutput(port[0])]
            self.add_flow(dp, prior_pmac, match, actions)
            self.logger.info(
                "  Swicth (%s) flow rule: Table:0 Priority:%d "
                + "[ETH_DST(%s, %s)] -> Port:%s",
                self.get_dpid_string(dp.id), prior_pmac, pmac, mask, port[0]
            )

    # -------------------------------------------------------------------------
    # Add rules to Core switch
    # -------------------------------------------------------------------------
    def add_core_rules(self, dp):
        # Add flow rule: ARP -> drop
        self.add_rule_arp(dp)
        # Add flow rule: table 0 miss -> drop
        self.add_rule_tablemiss(dp)

        # Add flow rules to Pods (Aggregate switches)
        ofp_parser = dp.ofproto_parser
        prior_pmac = self.PRIOR_PMAC
        for pod, port in list(self.ports_to_pod[dp.id].items()):
            # Get Pod PMAC wildcard
            pmac, mask = self.generate_pmac_wildcard(pod)

            # Ethernet destination PMAC wildcard -> port to Pod
            match = ofp_parser.OFPMatch(eth_dst=(pmac, mask))
            actions = [ofp_parser.OFPActionOutput(port)]
            self.add_flow(dp, prior_pmac, match, actions)
            self.logger.info(
                "  Swicth (%s) flow rule: Table:0 Priority:%d "
                + "[ETH_DST(%s, %s)] -> Port:%s",
                self.get_dpid_string(dp.id), prior_pmac, pmac, mask, port
            )

    # -------------------------------------------------------------------------
    # Create Group Table
    # -------------------------------------------------------------------------
    def add_rule_group(self, dp, group_id, ports):
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser

        # Action buckets
        buckets = []
        for port in ports:
            actions = [ofp_parser.OFPActionOutput(port)]
            bucket = ofp_parser.OFPBucket(1, ofp.OFPP_ANY,
                                          ofp.OFPG_ANY, actions)
            buckets.append(bucket)

        # Group modification message
        mod = ofp_parser.OFPGroupMod(
            dp, ofp.OFPGC_ADD, ofp.OFPGT_SELECT, group_id, buckets
        )
        dp.send_msg(mod)
        self.logger.info(
            "  Switch (%s) group rule: Group:%s SELECT Ports:%s",
            self.get_dpid_string(dp.id), group_id, ports
        )

    # -------------------------------------------------------------------------
    # Add flow rule for ARP
    # -------------------------------------------------------------------------
    def add_rule_arp(self, dp):
        # Match ARP by EtherType
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP)

        # Drop ARP action by default (Aggregate and Core switches)
        ofp = dp.ofproto
        action = "DROP"
        actions = []
        # Edge switch -> go to controller
        if self.lldp_level[dp.id] == self.LVL_EDGE:
            action = "CONTROLLER"
            actions = [ofp_parser.OFPActionOutput(ofp.OFPP_CONTROLLER)]

        # Add flow rule
        self.add_flow(dp, self.PRIOR_ARP, match, actions)
        self.logger.info(
            "  Switch (%s) flow rule: Table:0 Priority:%d [ARP] -> %s",
            self.get_dpid_string(dp.id), self.PRIOR_ARP, action
        )

    # -------------------------------------------------------------------------
    # Add flow rule for table-miss
    # -------------------------------------------------------------------------
    def add_rule_tablemiss(self, dp, in_table=0):
        # Match any for table-miss
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch()

        # Build drop action and add flow rule in corresponding table
        actions = []
        prior = self.PRIOR_TABLEMISS
        self.add_flow(dp, prior, match, actions, in_table)
        self.logger.info(
            "  Switch (%s) flow rule: Table:%d Priority:%d [MISS] -> DROP",
            self.get_dpid_string(dp.id), in_table, prior
        )

    # -------------------------------------------------------------------------
    # ARP packet handler
    # -------------------------------------------------------------------------
    def arp_packet_handler(self, msg, pkt):
        # Get packet data
        dp = msg.datapath
        ofp = dp.ofproto
        in_port = msg.match["in_port"]

        # Get Ethernet data
        pkt_eth = pkt.get_protocol(ethernet.ethernet)
        eth_dst = pkt_eth.dst
        eth_src = pkt_eth.src

        # Get ARP data
        pkt_arp = pkt.get_protocol(arp.arp)
        src_mac = pkt_arp.src_mac
        dst_mac = pkt_arp.dst_mac
        src_ip = pkt_arp.src_ip
        dst_ip = pkt_arp.dst_ip

        # Get time since last topology discovery
        timestamp = time.time() - self.init_time

        # TODO REMOVE
        if in_port == ofp.OFPP_CONTROLLER:
            print("\n--------------------------------------------------------")
            print("GOT ARP FROM CONTROLLER")
            print("--------------------------------------------------------\n")
            return

        # Check ARP packet comes from a host connected to an Edge switch
        from_edge = False
        for edge in self.edges:
            if dp.id == edge.id:
                from_edge = True
                break
        if not from_edge and in_port not in self.ports_e2a[dp.id]:
            return
        self.logger.info(
            "\nReceived ARP packet in Switch (%s), port %s: "
            + "%s (%s) -> %s (%s) | Ethernet: %s -> %s | Time: %s",
            self.get_dpid_string(dp.id), in_port, src_ip, src_mac, dst_ip,
            dst_mac, eth_src, eth_dst, self.get_time_string(timestamp)
        )

        # Generate PMAC from source host and add to routing table
        pmac = self.generate_pmac(dp.id, in_port, src_mac, src_ip)
        self.ip_table[src_ip] = {
            "dp": dp, "port": in_port, "amac": src_mac, "pmac": pmac
        }

        # Add flow rules for host MAC
        if (dp.id, src_mac, pmac) not in self.installed_rules:
            self.add_rule_pmac(dp, src_mac, pmac, in_port)
            self.installed_rules.append((dp.id, src_mac, pmac))

        # TODO DEVELOP
        # Add time to table
        # self.time_table[src_ip] = timestamp

        # Handle ARP request
        if pkt_arp.opcode == self.ARP_OPER_REQUEST:
            self.arp_request_handler(dp, in_port, src_ip, dst_ip, src_mac,
                                     dst_mac, eth_src, eth_dst, pmac, msg.data)
        # Handle ARP reply
        elif pkt_arp.opcode == 2:  # ARP reply
            self.arp_reply_handler(src_ip, pmac)

    # -------------------------------------------------------------------------
    # Generate Host PMAC
    # -------------------------------------------------------------------------
    def generate_pmac(self, dpid, port, mac, ip):
        # Get VM ID: check if a VM table already exists for (switch, port)
        vmid = self.INIT_VMID
        if (dpid, port) in self.vm_table:
            # Check if (MAC, IP) exists in VM table
            if (mac, ip) in self.vm_table[(dpid, port)]:
                # Get VM ID from (MAC, IP)
                vmid = self.vm_table[(dpid, port)][(mac, ip)]
            else:
                # Add (MAC, IP) to VM table
                vmid = len(self.vm_table[(dpid, port)])
                self.vm_table[(dpid, port)][(mac, ip)] = vmid
        else:
            # Create VM table for (switch, port) and add (MAC, IP)
            self.vm_table.setdefault((dpid, port), {})
            self.vm_table[(dpid, port)][(mac, ip)] = vmid

        # Get pod and position
        pod = self.td_edgepos[dpid]['pod']
        pos = self.td_edgepos[dpid]['pos']

        # Generate PMAC
        return "0a{pod:02x}{pos:02x}{port:02x}{vm:04x}".format(
            pod=pod, pos=pos, port=port, vm=vmid
        )

    # -------------------------------------------------------------------------
    # Add PMAC flow rules to Edge swicth
    # -------------------------------------------------------------------------
    def add_rule_pmac(self, dp, amac, pmac, port):
        # Table 0: source MAC -> source PMAC and table 1
        to_table = 1
        prior_pmac = self.PRIOR_PMAC
        ofp_parser = dp.ofproto_parser
        match = ofp_parser.OFPMatch(eth_src=amac)
        actions = [ofp_parser.OFPActionSetField(eth_src=pmac)]
        self.add_flow_table(dp, prior_pmac, to_table, match, actions)
        self.logger.info(
            "  Swicth (%s) flow rule: Table:0 Priority:%d "
            + "[ETH_SRC=%s] -> ETH_SRC=%s + Table:%s",
            self.get_dpid_string(dp.id), prior_pmac, amac, pmac, to_table
        )

        # Table 1: destination PMAC -> destination MAC and port
        in_table = 1
        match = ofp_parser.OFPMatch(eth_dst=pmac)
        actions = [
            ofp_parser.OFPActionSetField(eth_dst=amac),
            ofp_parser.OFPActionOutput(port)
        ]
        self.add_flow(dp, prior_pmac, match, actions, in_table)
        self.logger.info(
            "  Swicth (%s) flow rule: Table:%d Priority:%d "
            + "[ETH_DST=%s] -> ETH_DST=%s + Port:%s",
            self.get_dpid_string(dp.id), in_table, prior_pmac, pmac, amac, port
        )

    # -------------------------------------------------------------------------
    # ARP request handler
    # -------------------------------------------------------------------------
    def arp_request_handler(self, dp, in_port, src_ip, dst_ip, src_mac,
                            dst_mac, eth_src, eth_dst, src_pmac, data):
        # If destination IP is unknown by the controller
        if dst_ip not in self.ip_table:
            self.logger.info(
                "  ARP request to unknown IP: %s", dst_ip
            )
            self.flood_arp_request(dp, in_port, src_ip, dst_ip, dst_mac,
                                   eth_dst, src_pmac, data)
        # If destindation IP is known by the controller
        else:
            # Get PMAC associated to destination IP
            dst_pmac = self.ip_table[dst_ip]["pmac"]
            self.logger.info(
                "  ARP request to IP: %s -> PMAC: %s", dst_ip, dst_pmac
            )

            # TODO DEVELOP stale time of ARP (do not always flood)
            # if (self.time_table[src_ip] - self.time_table[dst_ip]) > _threshold:
            #     self.flood_arp_request(dp, in_port, src_ip, dst_ip, dst_mac,
            #                            eth_dst, pmac, data)

            # Reply to ARP request using known PMAC
            self.reply_arp_request(
                dp=dp, port=in_port, src_ip=dst_ip, dst_ip=src_ip,
                src_mac=dst_pmac, dst_mac=src_mac
            )

    # -------------------------------------------------------------------------
    # Send ARP request through all the ports of all Edge switches
    # -------------------------------------------------------------------------
    def flood_arp_request(self, dp, in_port, src_ip, dst_ip, dst_mac, eth_dst,
                          src_pmac, data):
        # Check if already flooded ARP request for destination IP
        if dst_ip in self.pending_reqs:
            # Check if source IP is already in pending request
            if src_ip not in self.pending_reqs[dst_ip]:
                self.pending_reqs[dst_ip].append(src_ip)
                self.logger.info(
                    "    Added IP %s to pending ARP request for IP %s",
                    src_ip, dst_ip
                )
            return

        # Add destination IP to pending ARP requests
        self.pending_reqs.setdefault(dst_ip, [])
        self.pending_reqs[dst_ip].append(src_ip)
        self.logger.info("    Added pending ARP request for IP %s from IP %s",
                         dst_ip, src_ip)

        # Rewrite ARP source MAC (arp_sha) with host PMAC and send through all
        # ports of the Edge switch
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        actions = [ofp_parser.OFPActionSetField(arp_sha=src_pmac),
                   ofp_parser.OFPActionOutput(ofp.OFPP_ALL)]
        out = ofp_parser.OFPPacketOut(
            datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
            in_port=in_port, actions=actions, data=data
        )
        dp.send_msg(out)

        # Send ARP request through all ports of the other Edge switches
        pkt = self.generate_arp_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_mac=src_pmac, dst_mac=dst_mac,
            eth_src=src_pmac, eth_dst=eth_dst, oper=self.ARP_OPER_REQUEST
        )
        for edge in self.edges:
            if edge.id != dp.id:
                actions = [ofp_parser.OFPActionOutput(ofp.OFPP_ALL)]
                out = ofp_parser.OFPPacketOut(
                    datapath=edge, buffer_id=ofp.OFP_NO_BUFFER,
                    in_port=ofp.OFPP_CONTROLLER, actions=actions,
                    data=pkt.data
                )
                edge.send_msg(out)
        self.logger.info("    Flooded ARP request through all Edge switches "
                         + "using PMAC %s as source MAC", src_pmac)

    # -------------------------------------------------------------------------
    # ARP reply handler
    # -------------------------------------------------------------------------
    def arp_reply_handler(self, src_ip, src_pmac):
        self.logger.info("  ARP reply from IP %s -> PMAC %s", src_ip, src_pmac)
        # Check if remain pending ARP requests for source IP
        if src_ip not in self.pending_reqs:
            self.logger.info("    No pending ARP requests for IP %s", src_ip)
            return

        # Reply to pending ARP requests for source IP
        dst_ips = list(self.pending_reqs[src_ip])
        for dst_ip in dst_ips:
            # Get destination data
            dst = self.ip_table[dst_ip]
            dp = dst["dp"]
            port = dst["port"]
            dst_mac = dst["amac"]

            # Send ARP reply packet
            self.reply_arp_request(
                dp=dp, port=port, src_ip=src_ip, dst_ip=dst_ip,
                src_mac=src_pmac, dst_mac=dst_mac
            )

    # -------------------------------------------------------------------------
    # Reply to an ARP request using the PMAC
    # -------------------------------------------------------------------------
    def reply_arp_request(self, dp, port, src_ip, dst_ip, src_mac, dst_mac):
        # Generate ARP reply packet using PMAC
        pkt = self.generate_arp_packet(
            src_ip=src_ip, dst_ip=dst_ip, src_mac=src_mac, dst_mac=dst_mac,
            eth_src=src_mac, eth_dst=dst_mac, oper=self.ARP_OPER_REPLY
        )

        # Send ARP reply packet
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        actions = [ofp_parser.OFPActionOutput(port)]
        out = ofp_parser.OFPPacketOut(
            datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
            in_port=ofp.OFPP_CONTROLLER, actions=actions, data=pkt.data
        )
        dp.send_msg(out)
        self.logger.info(
            "    Replied to ARP request from IP %s -> IP %s is in PMAC %s",
            dst_ip, src_ip, src_mac)

    # -------------------------------------------------------------------------
    # Add flow rule
    # -------------------------------------------------------------------------
    @staticmethod
    def add_flow(dp, prior, match, actions, in_table=0, idle_to=0, flags=0,
                 cookie=0):
        # Build rule instructions with apply actions
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        insts = [
            ofp_parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)
        ]

        # Send flow modification message
        mod = ofp_parser.OFPFlowMod(
            datapath=dp, priority=prior, match=match, instructions=insts,
            table_id=in_table, idle_timeout=idle_to, flags=flags, cookie=cookie
        )
        dp.send_msg(mod)

    # -------------------------------------------------------------------------
    # Add flow rule with table instructions
    # -------------------------------------------------------------------------
    @staticmethod
    def add_flow_table(dp, prior, to_table, match, actions=None, in_table=0):
        # Build rule instructions
        ofp = dp.ofproto
        ofp_parser = dp.ofproto_parser
        if actions:
            # Apply actions and go to table
            insts = [
                ofp_parser.OFPInstructionActions(
                    ofp.OFPIT_APPLY_ACTIONS, actions
                ),
                ofp_parser.OFPInstructionGotoTable(to_table)
            ]
        else:
            # Go to table
            insts = [ofp_parser.OFPInstructionGotoTable(to_table)]

        # Send flow modification message
        mod = ofp_parser.OFPFlowMod(
            datapath=dp, priority=prior, match=match,
            instructions=insts, table_id=in_table
        )
        dp.send_msg(mod)

    # -------------------------------------------------------------------------
    # Generate ARP packet
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_arp_packet(src_ip, dst_ip, src_mac, dst_mac,
                            eth_src, eth_dst, oper):
        eth_head = ethernet.ethernet(src=eth_src, dst=eth_dst,
                                     ethertype=ether_types.ETH_TYPE_ARP)
        arp_pkt = arp.arp(src_mac=src_mac, src_ip=src_ip, dst_mac=dst_mac,
                          dst_ip=dst_ip, opcode=oper)
        pkt = packet.Packet()
        pkt.add_protocol(eth_head)
        pkt.add_protocol(arp_pkt)
        pkt.serialize()
        return pkt

    # -------------------------------------------------------------------------
    # Generate PMAC wildcard
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_pmac_wildcard(pod=None, pos=None):
        # Check if matching pod
        pod_mask = "ff"
        if pod is None:
            pod = 0
            pod_mask = "00"

        # Check if matching pos
        pos_mask = "ff"
        if pos is None:
            pos = 0
            pos_mask = "00"

        return ("0a{pod:02x}{pos:02x}000000".format(pod=pod, pos=pos),
                "ff{pod}{pos}000000".format(pod=pod_mask, pos=pos_mask))

    # -------------------------------------------------------------------------
    # Get Datapath ID String
    # -------------------------------------------------------------------------
    @staticmethod
    def get_dpid_string(dpid):
        return "{dpid:016x}".format(dpid=dpid)

    # -------------------------------------------------------------------------
    # Get Time String
    # -------------------------------------------------------------------------
    @staticmethod
    def get_time_string(timestamp):
        return "%dd %02d:%02d\'%02d\"%03d" % (
            timestamp / 60 / 60 / 24, timestamp / 60 / 60 % 24,
            timestamp / 60 % 60, timestamp % 60, timestamp * 1000 % 1000
        )
