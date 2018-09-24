#!/usr/bin/env python

"""
This script is used to construct the graph samples for
input into the models.
"""

from __future__ import print_function
from __future__ import division

import os
import logging
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from trackml import dataset

from graph import construct_graph, save_graphs


def parse_args():
    parser = argparse.ArgumentParser('prepareData.py')
    add_arg = parser.add_argument
    add_arg('--input-dir',
            default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu10_pt1000_2017_07_29/')
            #default='/global/cscratch1/sd/sfarrell/ACTS/prod_mu200_pt500_2017_07_25')
    add_arg('--output-dir')
    add_arg('--n-files', type=int, default=1)
    add_arg('--n-workers', type=int, default=1)
    add_arg('--pt-min', type=float, default=1, help='pt cut')
    add_arg('--phi-slope-max', type=float, default=0.001,
            help='phi slope cut')
    add_arg('--z0-max', type=float, default=200, help='z0 cut')
    add_arg('--n-phi-sections', type=int, default=8,
            help='Break detector into number of phi sections')
    add_arg('--n-eta-sections', type=int, default=2,
            help='Break detector into number of eta sections')
    add_arg('--eta-range', type=float, nargs=2, default=[-5, 5],
            help='Considered range in eta')
    add_arg('--show-config', action='store_true',
            help='Dump the command line config')
    add_arg('--interactive', action='store_true',
            help='Drop into IPython shell at end of script')
    return parser.parse_args()

def select_hits(hits, truth, particles, pt_min=0):
    # Barrel volume and layer ids
    vlids = [(8,2), (8,4), (8,6), (8,8),
             (13,2), (13,4), (13,6), (13,8),
             (17,2), (17,4)]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    # Calculate particle transverse momentum
    pt = np.sqrt(particles.px**2 + particles.py**2)
    # True particle selection.
    # Applies pt cut, removes all noise hits.
    particles = particles[pt > pt_min]
    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id']], on='particle_id'))
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
    # Remove duplicate hits
    hits = hits.loc[
        hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
    ]
    return hits

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i+1]
        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]
        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j+1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))
    return hits_sections

def print_hits_summary(hits):
    """Log some summary info of the hits DataFrame"""
    n_events = hits.evtid.unique().shape[0]
    n_hits = hits.shape[0]
    n_particles = hits[['evtid', 'barcode']].drop_duplicates().shape[0]
    logging.info(('Hits summary: %i events, %i hits, %i particles,' +
                  ' %g particles/event, %g hits/event') %
                 (n_events, n_hits, n_particles,
                  n_particles/n_events, n_hits/n_events))

def process_event(prefix, pt_min, n_eta_sections, n_phi_sections,
                  eta_range, phi_range, phi_slope_max, z0_max):
    # Load the data
    evtid = int(prefix[-9:])
    logging.info('Event %i, loading data' % evtid)
    hits, particles, truth = dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])

    # Apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)

    # Divide detector into sections
    #phi_range = (-np.pi, np.pi)
    phi_edges = np.linspace(*phi_range, num=n_phi_sections+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections+1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)
    #hits_sectors = split_phi_sectors(hits, n_phi_sectors=n_phi_sectors)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sections, 1000.])

    # Define adjacent layers
    n_det_layers = 10
    l = np.arange(n_det_layers)
    layer_pairs = np.stack([l[:-1], l[1:]], axis=1)

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    graphs = [construct_graph(section_hits, layer_pairs=layer_pairs,
                              phi_slope_max=phi_slope_max, z0_max=z0_max,
                              feature_names=feature_names,
                              feature_scale=feature_scale)
              for section_hits in hits_sections]

    return graphs

def main():
    """Main program function"""
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')
    if args.show_config:
        logging.info('Command line config: %s' % args)

    # Construct layer pairs from adjacent layer numbers
    layers = np.arange(10)
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)

    # Find the input files
    all_files = os.listdir(args.input_dir)
    suffix = '-hits.csv'
    file_prefixes = sorted(os.path.join(args.input_dir, f.replace(suffix, ''))
                           for f in all_files if f.endswith(suffix))
    file_prefixes = file_prefixes[:args.n_files]

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, pt_min=args.pt_min,
                               n_eta_sections=args.n_eta_sections,
                               n_phi_sections=args.n_phi_sections,
                               eta_range=args.eta_range,
                               phi_range=(-np.pi, np.pi),
                               phi_slope_max=args.phi_slope_max,
                               z0_max=args.z0_max)
        graphs = pool.map(process_func, file_prefixes)

    # Merge across workers into one list of event samples
    graphs = [g for gs in graphs for g in gs]

    # Write outputs
    if args.output_dir:
        logging.info('Writing outputs to ' + args.output_dir)

        # Write out the graphs
        filenames = [os.path.join(args.output_dir, 'graph%06i' % i)
                     for i in range(len(graphs))]
        save_graphs(graphs, filenames)

    if args.interactive:
        import IPython
        IPython.embed()

    logging.info('All done!')

if __name__ == '__main__':
    main()
