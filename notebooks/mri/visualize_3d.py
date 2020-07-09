# %%
from ml4cvd.defines import MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.tensor_from_file import _mri_tensor_4d, _mri_hd5_to_structured_grids
import h5py
from mri_atria import to_xdmf

fpath = f'/mnt/disks/segmented-sax-lax/2020-07-07/2922335.hd5'

# %%
hd5 = h5py.File(fpath, 'r')
views_of_interest = [
    'cine_segmented_lax_2ch', 'cine_segmented_lax_3ch', 'cine_segmented_lax_4ch', 
    'cine_segmented_ao_dist', 'cine_segmented_lvot']

for view in views_of_interest:
    dss = _mri_hd5_to_structured_grids(hd5, view,
                                       view_name=view, 
                                       concatenate=False, save_path=None, order='F')
    to_xdmf(dss[0], f'2922335_{view}')



# %%
