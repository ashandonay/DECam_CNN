import h5py
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy import units as u

def TrainTest(dataset, md, cut, stop, value):

    hf = h5py.File(dataset, 'r')
    dataset_name = list(hf.keys())[0]
    images = np.array(hf.get(dataset_name))
    hf.close()
    X_train, X_test = images[0:cut], images[cut:stop]
    md_df = pd.read_csv(md)
    if 'OBJID_COPY' in md_df:
        md_df['OBJID'] = md_df['OBJID_COPY'].values
    md_train, md_test = md_df[0:cut], md_df[cut:stop]


    y_train, y_test = np.array([value] * len(X_train)), np.array([value] * len(X_test))

    return X_train, X_test, y_train, y_test, md_train, md_test

def gaussian2D(distance_to_center, sigma):
    return 1/(sigma**2*2*np.pi)*np.exp(-0.5*((distance_to_center)/sigma)**2)

def CenterFlux(images, metadata):
    search_images = images[:,0,:,:]
    psfs = metadata['PSF'].values
    psf_in_px = psfs / 0.263
    psf_in_px = psfs[:,np.newaxis,np.newaxis] / 0.263 / 2.3548
    yy, xx = np.indices(search_images[0].shape)
    center_x, center_y = 25, 25
    distance_to_center = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)[np.newaxis,:]
    distance_to_center = np.vstack(len(search_images)*[distance_to_center])
    psf_weights = gaussian2D(distance_to_center, psf_in_px)
    backgrounds = np.median(search_images, axis=(-1,-2))[:,np.newaxis,np.newaxis]
    center_fluxes = np.sum((search_images - backgrounds) * psf_weights, axis=(-1,-2))
    return center_fluxes

def SNR(metadata):
    metadata['SNR'] = metadata['FLUXCAL'].values/metadata['FLUXCALERR'].values
    return metadata['SNR'].values

def CatalogMatching(images, labels, metadata):
    # read in LightCurvesRealCoords.csv
    #snid_coords = pd.read_csv('LightCurvesRealCoords.csv')
    # assume metadata is stored in "md_df"
    #metadata_ = metadata.drop(['RA','DEC'],axis=1)
    #md_df = metadata_.merge(snid_coords, on='SNID', how='inner')
    # Load in catalog
    catalog_df = pd.read_csv('DES_Star_Catalog_SOF.csv')
    # Use astropy
    catalog_coords = SkyCoord(ra= catalog_df['STAR_RA'].values*u.degree, dec= catalog_df['STAR_DEC'].values * u.degree)
    candidate_coords = SkyCoord(ra= metadata['RA'].values*u.degree, dec= metadata['DEC'].values * u.degree)
    idx, d2d, d3d = match_coordinates_sky(candidate_coords, catalog_coords)
    #https://docs.astropy.org/en/stable/api/astropy.coordinates.match_coordinates_sky.html#astropy.coordinates.match_coordinates_sky
    # filter results
    threshold = 1 # in arcsec
    arc_sec_mask = (d2d > threshold * u.arcsec)
    md_indices = np.arange(len(metadata), dtype=int)
    keep_indices = md_indices[arc_sec_mask]
    md_objects_passing_catalog_match = metadata.loc[keep_indices]
    #apply to images
    return images[keep_indices], labels[keep_indices], md_objects_passing_catalog_match.reset_index(drop=True)

from scipy.stats import mode

def find_masks(ims, box_width=14):
    """
    Determine if there is masking over the transient in an array
    
    Args:
        ims (np.array): array of all images, shape = (N, 3, 51, 51)
        box_width (int): side length of center box to search for mask
        
    Returns:
        boolean array of length N, true if mask is detected
    """
    
    # Get the center of the difference image
    center = np.shape(ims[0])[-1] // 2
    box_width = box_width // 2
    center_boxes = ims[:, 2,
                       center - box_width : center + box_width, 
                       center - box_width : center + box_width]
    
    # Get the difference image mask values
    mask_vals = np.median(ims[:,2,:,:], axis=(-1, -2)).astype(int)
    
    # Get tht most common pixel values with a loop
    modes = np.array([mode(center_boxes[i], axis=None)[0] for i in range(len(center_boxes))])
    
    # Check when the mask value is the most common
    return modes.flatten() == mask_vals