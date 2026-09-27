from pathlib import Path
import hashlib
import json
import os
import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    os.replace(tmp, path)


def load_image(path):
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        image = tifffile.imread(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
    if image is None or image.ndim not in (2,3) or (image.ndim == 3 and image.shape[2] > 16):
        raise ValueError(f'{path}: expected a single YX or YXC plane (up to 16 channels)')
    if image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError(f'{path}: supported source dtypes are uint8, uint16, float32')
    if not np.isfinite(image).all(): raise ValueError(f'{path}: non-finite source pixels')
    return image


def preview(image, max_dim=1024):
    gray = image.astype(np.float32)
    if gray.ndim == 3: gray = gray.mean(axis=2)
    low, high = np.percentile(gray, [1,99])
    gray = np.clip((gray-low) / max(float(high-low), 1e-6),0,1)
    h,w = gray.shape
    scale = min(1., max_dim/max(h,w))
    small = cv2.resize(gray, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
    sx,sy = small.shape[1]/w,small.shape[0]/h
    return np.round(small*255).astype(np.uint8), dict(method='channel_mean, per-tile p1/p99 normalization, area resize', low=float(low), high=float(high), original_to_preview=[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])


def validate(manifest_path, out=None):
    source = Path(manifest_path).resolve(); raw=json.loads(source.read_text())
    if raw.get('schema_version') != '1.0': raise ValueError('Manifest schema_version must be 1.0')
    grid=raw.get('grid',{})
    if grid.get('row_direction','+y') != '+y' or grid.get('column_direction','+x') != '+x':
        raise ValueError('Only spatial rows +y and columns +x are currently supported')
    for key in ('nominal_overlap_fraction_x','nominal_overlap_fraction_y'):
        if grid.get(key) is not None and not 0<float(grid[key])<1:raise ValueError('Nominal overlap fractions must be between 0 and 1')
    tiles=[]; ids=set(); cells=set(); layout=None
    for original in raw.get('tiles',[]):
        tile=dict(original); id=tile.get('id'); row=tile.get('row'); col=tile.get('column')
        if not isinstance(id,str) or not id or id in ids: raise ValueError('Tile IDs must be unique nonempty strings')
        if type(row) is not int or type(col) is not int or (row,col) in cells: raise ValueError('Spatial row/column must be unique integer cells')
        ids.add(id);cells.add((row,col))
        for axis in ('x','y'):
            size=tile.get('pixel_size_um_'+axis)
            if size is not None and (not np.isfinite(float(size)) or float(size)<=0):raise ValueError('Pixel calibration must be finite and positive')
        path=(source.parent/tile['path']).resolve(); image=load_image(path)
        h,w=image.shape[:2]; channels=1 if image.ndim==2 else image.shape[2]
        for key,value in [('width',w),('height',h),('dtype',str(image.dtype))]:
            if key in tile and tile[key]!=value: raise ValueError(f'{id}: {key} does not match original file')
            tile[key]=value
        current=(str(image.dtype),channels,tuple(tile.get('channels',[f'channel_{i}' for i in range(channels)])),tile.get('timepoint'),tile.get('plane'),tile.get('z_index'),tile.get('pixel_size_um_x'),tile.get('pixel_size_um_y'))
        if layout is not None and layout!=current: raise ValueError('Separate incompatible channel/dtype planes into different manifests')
        layout=current
        channel_names=tile.get('channels',[f'channel_{i}' for i in range(channels)])
        if len(channel_names)!=channels: raise ValueError(f'{id}: channel count mismatch')
        tile.update(path=str(path),sha256=digest(path),channels=channel_names)
        mask=tile.get('mask')
        if mask:
            maskpath=(source.parent/mask).resolve(); m=load_image(maskpath)
            if m.shape!=(h,w): raise ValueError(f'{id}: mask must be a YX image with matching dimensions')
            tile.update(mask=str(maskpath),mask_sha256=digest(maskpath))
        tiles.append(tile)
    if not tiles: raise ValueError('Manifest contains no tiles')
    result={**raw,'tiles':tiles,'schema_version':'1.0','coordinate_contract':'pixel centers xy; d_ij=P_j-P_i; +x right +y down','source_manifest':str(source)}
    if out: save_json(out,result)
    return result


def neighbors(tiles):
    cells={(t['row'],t['column']):i for i,t in enumerate(tiles)}
    return [(i,cells[cell]) for i,t in enumerate(tiles) for cell in [(t['row'],t['column']+1),(t['row']+1,t['column'])] if cell in cells]


def demo_manifest(directory, selected=None):
    directory=Path(directory).resolve(); metadata=json.loads((directory/'positions.json').read_text())
    tiles=[]
    for t in metadata['tiles']:
        if selected is None or t['file'] in selected:
            tiles.append(dict(id=Path(t['file']).stem,path=str(directory/t['file']),row=t['row'],column=t['col']))
    return dict(schema_version='1.0',slide_id=directory.name,grid={'row_direction':'+y','column_direction':'+x'},tiles=tiles,notes='positions.json x/y are not treated as calibrated stage evidence; only spatial row/col used.')
