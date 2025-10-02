from utils import create_processor
from netsquid.nodes import Node, Connection, Network
from netsquid.qubits import QFormalism
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.models.qerrormodels import DepolarNoiseModel, FibreLossModel
from zalm_protocols import SPDCProcessProtocol, BeamSplitterProtocol, PBSProtocol, FilterProtocol, HeraldingProtocol, PostProcessorProtocol, MeasureStationProtocol, ZALMProtocol, SignalProtocol
import config
import numpy as np
#log = Logg()

def calculate_fiber_delay() -> float:
    """
    Calculates the propagation delay of a photon in an optical fiber.

    Returns:
        float: The propagation delay in nanoseconds.
    """

    speed_in_fiber = config.SPEED_OF_LIGHT_IN_VACUUM / config.REFRACTIVE_INDEX_OF_FIBER
    length_in_meters = config.INTERNODE_LENGTH * 1000
    delay_in_seconds = length_in_meters / speed_in_fiber
    delay_in_nanoseconds = delay_in_seconds * 1e9

    return delay_in_nanoseconds

def setup_fibre_loss():
    # Attenuation value for standard telecom fiber at 1550 nm (0.2 dB/km).
    # We convert this to a probability of loss per km.
    loss_db_per_km = 0.2
    #loss_prob_per_km = 1 - 10**(-loss_db_per_km / 10)
    #loss_prob_per_km = 10**(-loss_db_per_km / 10)
    loss_prob_per_km = (loss_db_per_km *np.log(10))/10
    #loss_prob_per_km = np.exp(-((loss_db_per_km *np.log(10))/10) * config.INTERNODE_LENGTH * 1000)
    # We can also model an initial coupling loss (e.g., 0.1 dB).
    coupling_loss_db = 0.1
    coupling_loss_prob = 1 - 10**(-coupling_loss_db / 10)

    # Create the loss model instance. This will be shared by all quantum channels.
    # We only apply this in REALISTIC mode.
    qchannel_models = None
    if True: #config.SIM_MODE == 'REALISTIC':
        fiber_loss_model = FibreLossModel(
            p_loss_init=0.0, #coupling_loss_prob,
            p_loss_length=loss_db_per_km
        )
        # A value of 1e3 Hz means significant scrambling over millisecond timescales,
        # which is realistic for long, unstabilized fibers.
        fiber_decoherence_model = DepolarNoiseModel(depolar_rate=1e-3, time_independent=True, qstate_formalism=QFormalism.DM)

        qchannel_models = {"quantum_loss_model": fiber_loss_model, "quantum_noise_model": fiber_decoherence_model}

    return qchannel_models

def network_setup(node_distance=4e-5, depolar_rate=0, dephase_rate=0.2):
    """Setup the physical components of the ZALM network.

    Parameters
    ----------
    node_distance : float, optional
        Distance between nodes.
    depolar_rate : float, optional
        Depolarization rate of qubits in memory.
    dephase_rate : float, optional
        Dephasing rate of physical measurement instruction.

    Returns
    -------
    :class:`~netsquid.nodes.node.Network`
        A Network with nodes "SPDCsource" and "NodeA",
        connected by an quantum connection and a classical connection

    """
    # Inter component distance is set at 1cm
    length = 10e-5  # km could also equal node_distance on instantiation
    delay_length = 4e-3 # 4 meters = 20ns based of SoL in fibre being 2x10^8 m/s
    mem_size = 4
    # Create a network of ZALM components
    network = Network(name="ZALM Source")

    SPDCa, SPDCb, BeamSplitter, PBSa, PBSb, FilterAH, FilterAV, FilterBH, FilterBV, HeraldingStation, PostProcessorA, PostProcessorB, MeasureStation, ZALM, Signal = network.add_nodes(nodes=[Node("SPDCa",
                                                qmemory=create_processor("SPDCa", 2)),
                                           Node("SPDCb",
                                                qmemory=create_processor("SPDCb", 2)),
                                           Node("BeamSplitter",
                                                qmemory=create_processor("BeamSplitter", 2)),
                                           Node("PBSa",
                                                qmemory=create_processor("PBSa", 2)),
                                           Node("PBSb",
                                                qmemory=create_processor("PBSb", 2)),
                                           Node("FilterAH",
                                                qmemory=create_processor("FilterAH", 2)),
                                           Node("FilterAV",
                                                qmemory=create_processor("FilterAV", 2)),
                                           Node("FilterBH",
                                                qmemory=create_processor("FilterBH", 2)),
                                           Node("FilterBV",
                                                qmemory=create_processor("FilterBV", 2)),
                                           Node("HeraldingStation",
                                                qmemory=create_processor("HeraldingStation", 2)),
                                           Node("PostProcessorA",
                                                qmemory=create_processor("PostProcessorA", 1)),
                                           Node("PostProcessorB",
                                                qmemory=create_processor("PostProcessorB", 1)),
                                           Node("MeasureStation",
                                                qmemory=create_processor("MeasureStation", 2)),
                                           Node("ZALM",
                                                 qmemory=create_processor("ZALM", 2)),
                                           Node("Signal",
                                                 qmemory=create_processor("Signal", 2))
                                                ])

    SPDCa.add_ports(["signal", "idler"])
    SPDCb.add_ports(["signal", "idler"])
    BeamSplitter.add_ports(["spdcaIN", "spdcbIN", "PBSaOUT", "PBSbOUT"])
    PBSa.add_ports(["qin", "qoutH", "qoutV"])
    PBSb.add_ports(["qin", "qoutH", "qoutV"])
    FilterAH.add_ports(["qin", "qout"])
    FilterAV.add_ports(["qin", "qout"])
    FilterBH.add_ports(["qin", "qout"])
    FilterBV.add_ports(["qin", "qout"])
    HeraldingStation.add_ports(["qinAH", "qinAV", "qinBH", "qinBV", "coutPPA", "coutPPB"])
    PostProcessorA.add_ports(["qin", "qout", "cHS"])
    PostProcessorB.add_ports(["qin", "qout", "cHS"])
    MeasureStation.add_ports(["qinA", "qinB"])

    network.add_connection(SPDCa, BeamSplitter,
                           channel_to=QuantumChannel("SPDCa_to_BeamSplitter",
                                                     length),
                           label="SPDCaBeamSplitter", port_name_node1="idler",
                           port_name_node2="spdcaIN")

    network.add_connection(SPDCb, BeamSplitter,
                            channel_to=QuantumChannel("SPDCb_to_BeamSplitter",
                                                         length),
                            label="SPDCbBeamSplitter", port_name_node1="idler",
                            port_name_node2="spdcbIN")

    network.add_connection(BeamSplitter, PBSa,
                            channel_to=QuantumChannel("BeamSplitter_to_PBSa",
                                                         length),
                            label="BeamSplitterPBSa", port_name_node1="PBSaOUT",
                            port_name_node2="qin")

    network.add_connection(BeamSplitter, PBSb,
                            channel_to=QuantumChannel("BeamSplitter_to_PBSb",
                                                         length),
                            label="BeamSplitterPBSb", port_name_node1="PBSbOUT",
                            port_name_node2="qin")

    network.add_connection(PBSa, FilterAH,
                            channel_to=QuantumChannel("PBSa_to_FilterAH",
                                                         length),
                            label="PBSaFilterAH", port_name_node1="qoutH",
                            port_name_node2="qin")
    network.add_connection(PBSa, FilterAV,
                            channel_to=QuantumChannel("PBSa_to_FilterAV",
                                                         length),
                            label="PBSaFilterAV", port_name_node1="qoutV",
                            port_name_node2="qin")
    network.add_connection(PBSb, FilterBH,
                            channel_to=QuantumChannel("PBSb_to_FilterBH",
                                                         length),
                            label="PBSbFilterBH", port_name_node1="qoutH",
                            port_name_node2="qin")
    network.add_connection(PBSb, FilterBV,
                            channel_to=QuantumChannel("PBSb_to_FilterBV",
                                                         length),
                            label="PBSbFilterBV", port_name_node1="qoutV",
                            port_name_node2="qin")
    network.add_connection(FilterAH, HeraldingStation,
                            channel_to=QuantumChannel("FilterAH_to_HeraldingStation",
                                                         length),
                            label="FilterAHHeraldingStation", port_name_node1="qout",
                            port_name_node2="qinAH")
    network.add_connection(FilterAV, HeraldingStation,
                            channel_to=QuantumChannel("FilterAV_to_HeraldingStation",
                                                         length),
                            label="FilterAVHeraldingStation", port_name_node1="qout",
                            port_name_node2="qinAV")
    network.add_connection(FilterBH, HeraldingStation,
                            channel_to=QuantumChannel("FilterBH_to_HeraldingStation",
                                                         length),
                            label="FilterBHHeraldingStation", port_name_node1="qout",
                            port_name_node2="qinBH")
    network.add_connection(FilterBV, HeraldingStation,
                            channel_to=QuantumChannel("FilterBV_to_HeraldingStation",
                                                         length),
                            label="FilterBVHeraldingStation", port_name_node1="qout",
                            port_name_node2="qinBV")
    network.add_connection(SPDCa, PostProcessorA,
                           channel_to=QuantumChannel("SPDCa_to_PostProcessorA",
                                                     delay_length),
                           label="SPDCaPostProcessorA", port_name_node1="signal",
                           port_name_node2="qin")
    network.add_connection(SPDCb, PostProcessorB,
                           channel_to=QuantumChannel("SPDCb_to_PostProcessorB",
                                                     delay_length),
                           label="SPDCbPostProcessorB", port_name_node1="signal",
                           port_name_node2="qin")
    network.add_connection(HeraldingStation, PostProcessorA, bidirectional=True,
                           channel_to=ClassicalChannel(
                               "HeraldingStationtoPPA", length),
                           channel_from=ClassicalChannel(
                               "PPAtoHeraldingStation", length),
                           label="HS_PPA_classical", port_name_node1="coutPPA",
                           port_name_node2="cHS")
    network.add_connection(HeraldingStation, PostProcessorB, bidirectional=True,
                           channel_to=ClassicalChannel(
                               "HeraldingStationtoPPB", length),
                           channel_from=ClassicalChannel(
                               "PPBtoHeraldingStation", length),
                           label="HS_PPB_classical", port_name_node1="coutPPB",
                           port_name_node2="cHS")

    internode_length = config.INTERNODE_LENGTH
    error_models = setup_fibre_loss()
    delay = calculate_fiber_delay()

    network.add_connection(PostProcessorA, MeasureStation,
                           channel_to=QuantumChannel("PostProcessorA_to_MeasureStation",
                                                     delay=delay, length=internode_length, models=error_models),
                           label="ProcessorAMeasureStation", port_name_node1="qout",
                           port_name_node2="qinA")
    network.add_connection(PostProcessorB, MeasureStation,
                           channel_to=QuantumChannel("PostProcessorB_to_MeasureStation",
                                                     delay=delay, length=internode_length, models=error_models),
                           label="PostProcessorBMeasureStation", port_name_node1="qout",
                           port_name_node2="qinB")

    """
    network.add_connection(PostProcessorA, MeasureStation,
                           channel_to=QuantumChannel("PostProcessorA_to_MeasureStation",
                                                     delay_length),
                           label="ProcessorAMeasureStation", port_name_node1="qout",
                           port_name_node2="qinA")
    network.add_connection(PostProcessorB, MeasureStation,
                           channel_to=QuantumChannel("PostProcessorB_to_MeasureStation",
                                                     delay_length),
                           label="PostProcessorBMeasureStation", port_name_node1="qout",
                           port_name_node2="qinB")
    """
    return network

def protocols_setup(network):
        #
        # Initialise the NodeProtocols
        #
        ZALM_protocol =     ZALMProtocol(network.get_node("ZALM"))
        Signal_protocol =   SignalProtocol(network.get_node("Signal")) # Only deals with signals
        SPDCa_protocol =    SPDCProcessProtocol(network.get_node("SPDCa"),name="a")
        SPDCb_protocol =    SPDCProcessProtocol(network.get_node("SPDCb"),name="b")
        BeamSplitter_protocol = BeamSplitterProtocol(network.get_node("BeamSplitter"))
        PBSa_protocol =     PBSProtocol(network.get_node("PBSa"))
        PBSb_protocol =     PBSProtocol(network.get_node("PBSb"))
        FilterAH_protocol = FilterProtocol(network.get_node("FilterAH"))
        FilterAV_protocol = FilterProtocol(network.get_node("FilterAV"))
        FilterBH_protocol = FilterProtocol(network.get_node("FilterBH"))
        FilterBV_protocol = FilterProtocol(network.get_node("FilterBV"))
        HS_protocol =       HeraldingProtocol(network.get_node("HeraldingStation"))
        PPA_protocol =      PostProcessorProtocol(network.get_node("PostProcessorA"))
        PPB_protocol =      PostProcessorProtocol(network.get_node("PostProcessorB"))
        MeasureStation_protocol = MeasureStationProtocol(network.get_node("MeasureStation"))
        #
        # Connect the NodeProtocols to each other
        #
        Signal_protocol.connect(zalm_protocol=ZALM_protocol, spdca_protocol=SPDCa_protocol, spdcb_protocol=SPDCb_protocol, bs_protocol=BeamSplitter_protocol, PBSa_protocol=PBSa_protocol, PBSb_protocol=PBSb_protocol, measure_protocol=MeasureStation_protocol, hs_protocol=HS_protocol, filterAH_protocol=FilterAH_protocol, filterAV_protocol=FilterAV_protocol, filterBH_protocol=FilterBH_protocol, filterBV_protocol=FilterBV_protocol, postA_protocol=PPA_protocol, postB_protocol=PPB_protocol)
        ZALM_protocol.connect(signal_protocol=Signal_protocol)
        BeamSplitter_protocol.connect(spdca_protocol=SPDCa_protocol, spdcb_protocol=SPDCb_protocol, PBSa_protocol=PBSa_protocol, PBSb_protocol=PBSb_protocol)
        SPDCa_protocol.connect(zalm_protocol=ZALM_protocol, bs_protocol=BeamSplitter_protocol, post_protocol=PPA_protocol)
        SPDCb_protocol.connect(zalm_protocol=ZALM_protocol, bs_protocol=BeamSplitter_protocol, post_protocol=PPB_protocol)
        PBSa_protocol.connect(bs_protocol=BeamSplitter_protocol, filterH_protocol=FilterAH_protocol, filterV_protocol=FilterAV_protocol, S_protocol=Signal_protocol)
        PBSb_protocol.connect(bs_protocol=BeamSplitter_protocol, filterH_protocol=FilterBH_protocol, filterV_protocol=FilterBV_protocol, S_protocol=Signal_protocol)
        FilterAH_protocol.connect(PBS_protocol=PBSa_protocol, HS_protocol=HS_protocol, S_protocol=Signal_protocol)
        FilterAV_protocol.connect(PBS_protocol=PBSa_protocol, HS_protocol=HS_protocol, S_protocol=Signal_protocol)
        FilterBH_protocol.connect(PBS_protocol=PBSb_protocol, HS_protocol=HS_protocol, S_protocol=Signal_protocol)
        FilterBV_protocol.connect(PBS_protocol=PBSb_protocol, HS_protocol=HS_protocol, S_protocol=Signal_protocol)
        HS_protocol.connect(filterAH_protocol=FilterAH_protocol, filterAV_protocol=FilterAV_protocol, filterBH_protocol=FilterBH_protocol, filterBV_protocol=FilterBV_protocol, postA_protocol=PPA_protocol, postB_protocol=PPB_protocol)
        PPA_protocol.connect(spdc_protocol=SPDCa_protocol, HS_protocol=HS_protocol, signal_protocol=Signal_protocol)
        PPB_protocol.connect(spdc_protocol=SPDCb_protocol, HS_protocol=HS_protocol, signal_protocol=Signal_protocol)
        MeasureStation_protocol.connect(S_protocol=Signal_protocol)

        protocols = [ZALM_protocol, Signal_protocol, SPDCa_protocol, SPDCb_protocol, BeamSplitter_protocol, PBSa_protocol, PBSb_protocol, FilterAH_protocol, FilterAV_protocol, FilterBH_protocol, FilterBV_protocol, HS_protocol, PPA_protocol, PPB_protocol, MeasureStation_protocol]

        return protocols

def cleanup(network, protocols):
    for protocol in protocols:
        protocol.remove()

    nodes = []
    for node in network.nodes.values():
        nodes.append(node.name)
    for node in nodes:
        network.remove_node(node)
    network = None
