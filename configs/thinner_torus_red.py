from lib.engine import MPMSimulator

_base_ = '../configs/default.py'

cfg = dict(
    dtype='float32',
    n_cameras=21,
    n_frames=20,
    xyz_min=[-0.5, 0.1, -0.5],
    xyz_max=[0.5, 1.2, 0.5],
    dx=0.16,
    material=MPMSimulator.elasticity,
    pg_scale=[1000, 2000, 4000],
    data_dir='data/thinner_torus_red',
    base_dir='checkpoint/thinner_torus_red',
    cuda_chunk_size=100,
    nerf_bs=2 ** 17,
    particle_chunk_size=2 ** 10,
    taichi_cuda_memory=0.5,
    # taichi_cuda_memory=0.3,
    dt=1 / 24 / 200,
    H=800,
    W=800,
    N_static=6001,
    # N_static = 2001,
    entropy_weight=1e-3,
    volume_weight=1e-3,
    # tv_weight = 1e-4,
    direct_nerf=False,
    E=1e3,
    nu=0.1,
    gt_E=1e4,
    gt_nu=0.25,
    rho=1000,
    hit_frame=5,
    physical_params=dict(global_E=1e-1, global_nu=1e-2),
    write_out=True,
    random_particles=True,
)

del MPMSimulator
