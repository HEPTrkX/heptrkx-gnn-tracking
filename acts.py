"""
Helper code for parsing and using the ACTS data.
The code in this module depends on pandas.
"""

from __future__ import print_function

import ast
import multiprocessing as mp

import pandas as pd
import numpy as np

def load_data_events(file_name, columns, start_evtid=0):
    """
    Load data from file into a pandas dataframe.
    
    Uses python's ast parser to extract the nested list structures.
    This implementation assumes there is no event ID saved in the file
    and that it must detect events based on the presence of blank lines.
    """
    dfs = []
    print('Loading', file_name)
    with open(file_name) as f:
        event_lines = []
        # Loop over lines in the file
        for line in f:
            # Add to current event
            if line.strip() and line[0] != '#':
                event_lines.append(ast.literal_eval(line))
            
            # Finalize a complete event
            elif len(event_lines) > 0:
                evtid = len(dfs) + start_evtid
                df = pd.DataFrame(event_lines)
                df.columns = columns
                df['evtid'] = evtid
                dfs.append(df)
                event_lines = []
        # Verify there are no leftovers (otherwise fix this code)
        assert len(event_lines) == 0
    
    # Concatenate the events together into one DataFrame
    return pd.concat(dfs, ignore_index=True)

def process_hits_data(df, copy_keys=['evtid', 'barcode', 'volid', 'layid']):
    """Parse out the columns and calculate derived variables"""
    x = df.gpos.apply(lambda pos: pos[0]).astype(np.float32)
    y = df.gpos.apply(lambda pos: pos[1]).astype(np.float32)
    z = df.gpos.apply(lambda pos: pos[2]).astype(np.float32)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return df[copy_keys].assign(z=z, r=r, phi=phi)

def process_particles_data(df, copy_keys=['evtid', 'barcode', 'q']):
    """Parse out the columns and calculate derived variables"""
    p = df.kin.apply(lambda kin: kin[0]).astype(np.float32)
    theta = df.kin.apply(lambda kin: kin[1]).astype(np.float32)
    phi = df.kin.apply(lambda kin: kin[2]).astype(np.float32)
    pt = p * np.sin(theta)
    eta = -1. * np.log(np.tan(theta / 2.))
    return df[copy_keys].assign(pt=pt, eta=eta, phi=phi)

def read_worker(hits_file):
    """DEPRECATED; use read_hits_worker"""
    hits_columns = ['hitid', 'barcode', 'volid', 'layid', 'lpos',
                    'lerr', 'gpos', 'chans', 'dir', 'direrr']
    return process_hits_data(load_data_events(hits_file, columns=hits_columns))

def process_files_deprecated(hits_files, num_workers, concat=True):
    """Load and process a set of hits files with MP"""
    pool = mp.Pool(processes=num_workers)
    hits = pool.map(read_worker, hits_files)
    pool.close()
    pool.join()
    # Fix the evtid to be consecutive
    for i in range(1, len(hits)):
        hits[i].evtid += hits[i-1].evtid.iloc[-1] + 1
    if concat:
        return pd.concat(hits, ignore_index=True)
    else:
        return hits

def read_hits_worker(hits_file):
    columns = ['hitid', 'barcode', 'volid', 'layid', 'lpos',
               'lerr', 'gpos', 'chans', 'dir', 'direrr']
    return process_hits_data(load_data_events(hits_file, columns=columns))

def read_particles_worker(particles_file):
    columns = ['barcode', 'vpos', 'kin', 'q']
    return process_particles_data(load_data_events(particles_file, columns=columns))

def process_hits_files(hits_files, pool):
    """Load and process a set of hits files with MP"""
    hits = pool.map(read_hits_worker, hits_files)
    # Fix the evtid to be consecutive
    for i in range(1, len(hits)):
        hits[i].evtid += hits[i-1].evtid.iloc[-1] + 1
    return hits

def process_particles_files(particles_files, pool):
    """Load and process a set of particles files with MP"""
    particles = pool.map(read_particles_worker, particles_files)
    # Fix the evtid to be consecutive
    for i in range(1, len(particles)):
        particles[i].evtid += particles[i-1].evtid.iloc[-1] + 1
    return particles

def process_files(hits_files, particles_files, pool):
    """Load and process a set of hits and particles files with MP"""
    hits = process_hits_files(hits_files, pool)
    particles = process_particles_files(particles_files, pool)
    return hits, particles

def select_barrel_hits(hits):
    """Selects hits in the barrel volumes and re-enumerates volumes and layers"""
    # Select all barrel hits
    vids = [8, 13, 17]
    barrel_hits = hits[np.logical_or.reduce([hits.volid == v for v in vids])]
    # Re-enumerate the volume and layer numbers for convenience
    volume = pd.Series(-1, index=barrel_hits.index, dtype=np.int8)
    vid_groups = barrel_hits.groupby('volid')
    for i, v in enumerate(vids):
        volume[vid_groups.get_group(v).index] = i
    # This assumes 4 layers per volume (except last volume)
    layer = (barrel_hits.layid / 2 - 1 + volume * 4).astype(np.int8)
    return (barrel_hits[['evtid', 'barcode', 'phi', 'z']]
            .assign(volume=volume, layer=layer))

def bin_barrel_hits(hits, evtids, vols, bins, ranges):
    """Construct the per-volume images by binning the hits"""
    # Prepare to lookup hits by evtid
    evt_groups = hits.groupby('evtid')
    # Unique event IDs
    if evtids is None:
        evtids = hits.evtid.drop_duplicates().values
    n_samples = evtids.shape[0]
    # Construct the empty data
    n_vols = len(vols)
    hists = [np.zeros([n_samples] + bins[iv], dtype=np.uint16)
             for iv in range(n_vols)]
    # Loop over events
    for i, evtid in enumerate(evtids):
        # Get the hits for this event
        evt_hits = evt_groups.get_group(evtid)
        # Loop over volumes
        for iv in range(n_vols):
            vol = vols[iv]
            # Get the hits for this volume
            vol_hits = evt_hits[evt_hits.volume == vol]
            # Bin the hits
            hdata = (vol_hits.layer.values, vol_hits.phi.values, vol_hits.z.values)
            hists[iv][i] = np.histogramdd(hdata, bins=bins[iv], range=ranges[iv])[0]
    return hists

def data_consistent(h, p):
    join_keys = ['evtid', 'barcode']
    matches = p[join_keys].merge(h[join_keys], on=join_keys)
    valid = (matches.shape[0] == h.shape[0])
    if not valid:
        print('Invalid data file found!')
    return valid

def check_data_consistency(hits, particles):
    """
    Make sure every hit has a particle in the corresponding truth data.
    Inputs hits and particles are lists of dataframes from input files.
    """
    data = [(h,p) for (h,p) in zip(hits, particles) if data_consistent(h, p)]
    hits = [hp[0] for hp in data]
    particles = [hp[1] for hp in data]
    return hits, particles