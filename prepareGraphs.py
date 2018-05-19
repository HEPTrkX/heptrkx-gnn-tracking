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
    add_arg('--n-phi-sectors', type=int, default=8,
            help='Break detector into number of phi sectors')
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

def split_phi_sectors(hits, n_phi_sectors=8):
    phi_sector_width = 2 * np.pi / n_phi_sectors
    phi_sector_edges = np.linspace(-np.pi, np.pi, n_phi_sectors + 1)
    hits_sectors = []
    # Loop over phi sectors
    for i in range(n_phi_sectors):
        phi_sector_min, phi_sector_max = phi_sector_edges[i:i+2]
        # Select hits from this sector
        sector_hits = hits[(hits.phi > phi_sector_min) & (hits.phi < phi_sector_max)]
        # Center the phi sector on 0
        centered_phi = sector_hits.phi - phi_sector_min - phi_sector_width / 2
        sector_hits = sector_hits.assign(phi=centered_phi, phi_sector=i)
        hits_sectors.append(sector_hits)
    # Return results
    return hits_sectors

def print_hits_summary(hits):
    """Log some summary info of the hits DataFrame"""
    n_events = hits.evtid.unique().shape[0]
    n_hits = hits.shape[0]
    n_particles = hits[['evtid', 'barcode']].drop_duplicates().shape[0]
    logging.info(('Hits summary: %i events, %i hits, %i particles,' +
                  ' %g particles/event, %g hits/event') %
                 (n_events, n_hits, n_particles,
                  n_particles/n_events, n_hits/n_events))

def process_event(prefix, pt_min, n_phi_sectors, phi_slope_max, z0_max):
    # Load the data
    evtid = int(prefix[-9:])
    logging.info('Event %i, loading data' % evtid)
    hits, particles, truth = dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])

    # Apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)
    hits_sectors = split_phi_sectors(hits, n_phi_sectors=n_phi_sectors)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sectors, 1000.])

    # Define adjacent layers
    n_det_layers = 10
    l = np.arange(n_det_layers)
    layer_pairs = np.stack([l[:-1], l[1:]], axis=1)

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    graphs = [construct_graph(sector_hits, layer_pairs=layer_pairs,
                              phi_slope_max=phi_slope_max, z0_max=z0_max,
                              feature_names=feature_names,
                              feature_scale=feature_scale)
              for sector_hits in hits_sectors]

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
                               n_phi_sectors=args.n_phi_sectors,
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
