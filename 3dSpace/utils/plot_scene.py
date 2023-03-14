import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from vispy import app, visuals, scene
from matrixGridMap import *

def plot_girdMap(gridMap: matrixGridMap3D):
    """
    plot the grid map
    """
    vol = gridMap.get_data().astype(np.int16) * 100
    vol = vol.transpose((2,1,0))

    canvas = scene.SceneCanvas(keys='interactive', title='Grid Map 3D', show=True)
    view = canvas.central_widget.add_view()
    volume = scene.visuals.Volume(vol, clim=(0,200), parent=view.scene, method='mip')
    cam = scene.cameras.TurntableCamera(parent=view.scene, name='Turntable')
    view.camera = cam

    canvas.app.run()

def show_gridMap_with_bullet(gridMap: matrixGridMap3D, pclient):
    import pybullet as p
    import pybullet_data
    """
    Load obstacles into the pybullet simulation environment
    :param pclient: pybullet client
    :param obstacles: dict of obstacles, temporarily support:
        {type: "box", center(1x3 array), half_extend(1x3 array), yaw_rad, fixed(bool)}
        {type: "cylinder", center(1x3 array), radius, height, fixed(bool)}
        {type: "ball", center(1x3 array), radius, fixed(bool)}
        {type: "height_field", scale(1x3 array), grid_size(1x2 array), height_data(MxN)}(seperatly loaded)
    """
    if p.getNumBodies(physicsClientId=pclient) == 0:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=pclient)
    
    uid_list = []
    only_height_field = None
    obstacles = gridMap.get_obs_dict()
    for (_, param) in obstacles.items():
        if param['type'] == "box":
            direction_quat = p.getQuaternionFromEuler([0., 0., param['yaw_rad']])
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                halfExtents=param['half_extend'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=param['half_extend'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                baseOrientation=direction_quat,
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == "cylinder":
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                                radius=param['radius'],
                                                length=param['height'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=param['radius'],
                                                height=param['height'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == 'ball':
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                radius=param['radius'],
                                                physicsClientId=pclient)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                radius=param['radius'],
                                                physicsClientId=pclient)
            uid = p.createMultiBody(baseMass=0 if param['fixed'] else 1.,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=param['center'],
                                useMaximalCoordinates=True,
                                physicsClientId=pclient)
        elif param['type'] == "height_field":
            field = param['height_field']
            data = field.get_height_data()
            if only_height_field is None:
                terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                    meshScale=param['scale'],
                                                    heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                    heightfieldData=param['height_data'].flatten(order='F'),
                                                    numHeightfieldRows=data.shape[0],
                                                    numHeightfieldColumns=data.shape[1],
                                                    physicsClientId=pclient)
                only_height_field = terrainShape
            else:
                terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                    meshScale=param['scale'],
                                                    heightfieldTextureScaling=(data.shape[0]-1)/2,
                                                    heightfieldData=param['height_data'],
                                                    numHeightfieldRows=data.shape[0],
                                                    numHeightfieldColumns=data.shape[1],
                                                    replaceHeightfieldIndex=only_height_field,
                                                    physicsClientId=pclient)
                continue
            uid = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=terrainShape,
                                physicsClientId=pclient)
            pos_bias = [data.shape[0]*param['scale'][0] / 2,
                        data.shape[1]*param['scale'][1] / 2,
                        0]
            p.resetBasePositionAndOrientation(uid, pos_bias, [0,0,0,1], physicsClientId=pclient)
            p.changeVisualShape(uid, -1, rgbaColor=[1,1,1,1], physicsClientId=pclient)
        else:
            print("Warning: undefined obstacle type! Loading nothing!")
            uid = -1
        uid_list.append(uid)
    return uid_list


if __name__ == "__main__":
    import os
    import time
    from heightField import constructHeightFieldFromImg

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/pictures/test.png'
    field = constructHeightFieldFromImg(path, 3.0, [0,0], cell_size=0.03, data_shape=(60,70))
    obs_dict = {'obs1': {'type': 'height_field', 'height_field':field}}

    start_time = time.time()
    map = matrixGridMap3D(center_crd=[0,0,0], half_extend=[3,3,2], cell_size=0.05)
    cons_time = time.time()
    print('construction time: ', cons_time - start_time)
    map.add_obstacles(obs_dict=obs_dict)
    load_time = time.time()
    print('load time: ', load_time - cons_time)

    plot_girdMap(map)