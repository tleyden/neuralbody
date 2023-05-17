import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    print("make_network is loading network from: ", path)
    network = imp.load_source(module, path).Network()
    return network
