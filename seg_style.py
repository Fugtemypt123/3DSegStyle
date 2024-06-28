import torch
import numpy as np
import os
import viser
import viser.transforms as tf
from collections import deque
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from scene.VGG import VGGEncoder, normalize_vgg
from scene import Scene
from gaussian_renderer import render
from scene.cameras import Camera
from utils.general_utils import get_image_paths
import SegAnyGAussians.scene_seg as SegAny
from hdbscan import HDBSCAN
from sklearn.neighbors import NearestNeighbors


class ViserViewer:
    def __init__(self, gaussians, pipeline, background, override_color, training_cams, wikiart_img_paths=None, viewer_port='8080'):
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.override_color = override_color
        self.fov = [training_cams[0].FoVx, training_cams[0].FoVy]

        self.port = viewer_port

        self.render_times = deque(maxlen=3)

        self.vgg_encoder = VGGEncoder().cuda()

        self.display_interpolation = False
        self.style_img = None

        # Set up the server, init GUI elements
        self.server = viser.ViserServer(port=self.port)
        self.need_update = False

        with self.server.add_gui_folder("Rendering Settings"):
            self.reset_view_button = self.server.add_gui_button("Reset View")

            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=2048, step=5, initial_value=1024
            )

            self.jpeg_quality_slider = self.server.add_gui_slider(
                "JPEG Quality", min=0, max=100, step=1, initial_value=80
            )

            self.training_view_slider = self.server.add_gui_slider(
                "Training View",
                min=0,
                max=len(training_cams) - 1,
                step=1,
                initial_value=0,
            )

            self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)


        with self.server.add_gui_folder("Style Transfer"):
            self.style_img_path_text = self.server.add_gui_text(
                    "Style Image",
                    initial_value="",
                    hint="Path to style image",                
                )
            
            self.display_style_img = self.server.add_gui_checkbox("Display Style Image", initial_value=True)

            if wikiart_img_paths is not None:
                self.random_style_button = self.server.add_gui_button("Random Style")

        
        with self.server.add_gui_folder("Style Interpolation"):
            self.style_path_1 = self.server.add_gui_text(
                "Style 1",
                initial_value="",
                hint="Path to style image",
            )

            self.style_path_2 = self.server.add_gui_text(
                "Style 2",
                initial_value="",
                hint="Path to style image",
            )

            self.interpolation_ratio = self.server.add_gui_slider(
                "Interpolation Ratio",
                min=0,
                max=1,
                step=0.01,
                initial_value=0.5,
            )


        # Handle GUI events
        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.jpeg_quality_slider.on_update
        def _(_):
            self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        @self.training_view_slider.on_update
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                target_camera = training_cams[self.training_view_slider.value]
                target_R = target_camera.R
                target_T = target_camera.T

                with client.atomic():
                    client.camera.wxyz = tf.SO3.from_matrix(target_R).wxyz
                    client.camera.position = -target_R @ target_T
                    self.fov = [target_camera.FoVx, target_camera.FoVy]
                    client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                        [0.0, -1.0, 0.0]
                    )

        @self.style_img_path_text.on_update
        def _(_):
            self.need_update = True
            self.display_interpolation = False
            style_img_path = self.style_img_path_text.value
            # read style image and extract features
            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
            style_img_features = self.vgg_encoder(normalize_vgg(style_img))
            self.style_img = resize(style_img, (128,128))

            # style transfer
            tranfered_features = self.gaussians.style_transfer(
                self.gaussians.final_vgg_features.detach(), # point cloud features [N, C]
                style_img_features.relu3_1,
            )
            self.override_color = self.gaussians.decoder(tranfered_features) # [N, 3]

        @self.display_style_img.on_update
        def _(_):
            self.need_update = True

        if wikiart_img_paths is not None:
            @self.random_style_button.on_click
            def _(_):
                self.need_update = True
                style_img_path = np.random.choice(wikiart_img_paths)
                self.style_img_path_text.value = style_img_path

        @self.style_path_1.on_update
        def _(_):
            style_interpolation()

        @self.style_path_2.on_update
        def _(_):
            style_interpolation()

        @self.interpolation_ratio.on_update
        def _(_):
            style_interpolation()

        def style_interpolation():
            if not self.style_path_1.value or not self.style_path_2.value:
                return
            
            self.need_update = True

            style_path_1 = self.style_path_1.value
            style_path_2 = self.style_path_2.value

            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img0 = trans(Image.open(style_path_1)).cuda()[None, :3, :, :]
            style_img1 = trans(Image.open(style_path_2)).cuda()[None, :3, :, :]
        
            style_img_features0 = self.vgg_encoder(normalize_vgg(style_img0))
            style_img_features1 = self.vgg_encoder(normalize_vgg(style_img1))
            self.style_img0 = resize(style_img0, (128,128))
            self.style_img1 = resize(style_img1, (128,128))

            tranfered_features0 = gaussians.style_transfer(
                gaussians.final_vgg_features.detach(), 
                style_img_features0.relu3_1,
            )
            tranfered_features1 = gaussians.style_transfer(
                gaussians.final_vgg_features.detach(), 
                style_img_features1.relu3_1,
            )

            interpolated_features = (1-self.interpolation_ratio.value)*tranfered_features0 + self.interpolation_ratio.value*tranfered_features1

            self.override_color = self.gaussians.decoder(interpolated_features) # [N, 3]

            self.display_interpolation = True

        self.label_to_color = torch.rand(1000, 3).cuda()

    @torch.no_grad()
    def update(self):
        # if self.need_update and self.override_color is not None:
        if self.need_update:
            interval = None
            for client in self.server.get_clients().values():
                camera = client.camera
                R = tf.SO3(camera.wxyz).as_matrix()
                T = -R.T @ camera.position

                # get camera poses
                W = self.resolution_slider.value
                H = int(self.resolution_slider.value/camera.aspect)

                view = Camera(
                    colmap_id=None,
                    R=R,
                    T=T,
                    FoVx=self.fov[0],
                    FoVy=self.fov[1],
                    image=None, 
                    gt_alpha_mask=None,
                    image_name=None,
                    uid=None,
                )
                view.image_height = H
                view.image_width = W

                start_cuda = torch.cuda.Event(enable_timing=True)
                end_cuda = torch.cuda.Event(enable_timing=True)
                start_cuda.record()

                # seg_colors = self.label_to_color[self.gaussians._cls] 
                # seg_colors = None

                # rendering = render(view, self.gaussians, self.pipeline, self.background, override_color=seg_colors)["render"]

                # 0: 8983 垃圾
                # 1: 2241 垃圾
                # 2: 1921 垃圾
                # 3: 3590 门边上地道那一侧的花盆以及旁边那一坨草
                # 4: 2555 基本是垃圾
                # 5: 2464
                # 6: 2861
                # 7: 2455
                # 8: 2133
                # 9: 2827
                # 10: 30365 有地道那面的墙
                # 11: 3759
                # 12: 8154 门另一侧花盆边上的草
                # 13: 6243 某一堆垃圾草
                # 14: 2191 垃圾
                # 15: 5878 垃圾草
                # 16: 6296 门边上一圈垃圾
                # 17: 3970 离桌子最近的那一棵树
                # 18: 3208 垃圾
                # 19: 3532 门的上半部分
                # 20: 4551 花！花！花！
                # 21: 26262 大概率是桌子，小概率是垃圾
                # 22: 3948 垃圾
                # 23: 34456 房子
                # 24: 59025 桌子下面的地板和桌子腿
                # 25: 4439 纯粹的垃圾
                # 26: 4186 垃圾
                # 27: 16777 桌子旁边挨着花盆的那一坨草
                # 28: 59140 桌子下面的地板外侧那一圈草
                rendering = render(view, self.gaussians, self.pipeline, self.background, override_color=self.override_color, index=20)["render"]
                rendering = rendering.clamp(0, 1)
                if self.display_style_img.value:
                    if not self.display_interpolation and self.style_img is not None:
                        rendering[:, -128:, -128:] = self.style_img.squeeze(0)
                    elif self.style_path_1.value and self.style_path_2.value:
                        rendering[:, -128:, -128:] = self.style_img1.squeeze(0)
                        rendering[:, -128:, :128] = self.style_img0.squeeze(0)
                    
                end_cuda.record()
                torch.cuda.synchronize()
                interval = start_cuda.elapsed_time(end_cuda)/1000.

                out = rendering.permute(1,2,0).cpu().numpy().astype(np.float32)
                client.set_background_image(out, format="jpeg", jpeg_quality=self.jpeg_quality_slider.value)

            if interval:
                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            else:
                self.fps.value = "NA"


def cluster_in_3D(gaussians, gate, scale, path):
    point_features = gaussians.get_point_features
    gates = gate(torch.tensor([scale]).cuda())

    scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * gates.unsqueeze(0)
    normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)
    sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]
    normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)

    clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)
    cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
    cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
    for i in range(0, len(np.unique(cluster_labels))):
        cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

    seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
    cluster_point = seg_score.argmax(dim = -1).cpu().numpy()
    np.save(path, cluster_point)
    # print("cluster_point_colors shape: ", cluster_point.shape)


def update_gaussian(gaussian, seg, style_gaussians):
    gaussian._cls = seg["cls"]
    ori_xyz = gaussian._xyz.cpu().numpy()
    sty_xyz = style_gaussians._xyz.cpu().numpy()
    sty_fet = style_gaussians.final_vgg_features
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(sty_xyz)
    _, indices = neigh.kneighbors(ori_xyz)
    indices = indices.flatten()
    ori_fet = sty_fet[indices]
    gaussian.final_vgg_features = ori_fet
    gaussian.decoder = style_gaussians.decoder
    gaussian.style_transfer = style_gaussians.style_transfer


@torch.no_grad()
def run_seg():
    # hyperparameters
    sh_degree = 3
    FEATURE_DIM = 32
    Scale = 1.0
    MODEL_PATH = '/network_space/server128/shared/yinshaofeng/StyleGaussian/SegAnyGAussians/pretrained_models/garden' 
    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000
    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')
    SAVE_CLS_PATH = "/network_space/server128/shared/yinshaofeng/StyleGaussian/SegAnyGAussians/pretrained_models/garden/point_cloud/iteration_10000/cluster_point.npy"
    
    # load segmentation gaussian model
    seg_gaussians = SegAny.GaussianModel(sh_degree)
    seg_fet_gaussians = SegAny.FeatureGaussianModel(FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()

    seg_gaussians.load_ply(SCENE_PCD_PATH)

    if not os.path.exists(SAVE_CLS_PATH):
        print("Clustering in 3D space...")
        seg_fet_gaussians.load_ply(FEATURE_PCD_PATH)
        scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
        cluster_in_3D(seg_fet_gaussians, scale_gate, Scale, SAVE_CLS_PATH)

    cluster_point = np.load(SAVE_CLS_PATH)
    seg = {
        "xyz": seg_gaussians._xyz,
        # "fet_dc": seg_gaussians._features_dc,
        # "fet_rest": seg_gaussians._features_rest,
        "cls": cluster_point
    }
    return seg


@torch.no_grad()
def run_viewer(dataset : ModelParams, pipeline : PipelineParams, wikiartdir, viewer_port):
    seg = run_seg()

    wikiart_img_paths = None
    if wikiartdir and os.path.exists(wikiartdir):
        print('Loading style images folder for random style transfer')
        wikiart_img_paths = get_image_paths(wikiartdir)

    # load trained gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # load style gaussian model
    load_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
    style_gaussians = GaussianModel(dataset.sh_degree)
    style_gaussians.restore(torch.load(load_path), from_style_model=True, xyz=scene.gaussians._xyz)

    # update original gaussian model
    update_gaussian(gaussians, seg, style_gaussians)

    # run viewer
    gui = ViserViewer(gaussians, pipeline, background, override_color=None, training_cams=scene.getTrainCameras(), wikiart_img_paths=wikiart_img_paths, viewer_port=viewer_port)
    while(True):
        gui.update()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--style_folder", type=str, default="images")
    parser.add_argument("--viewer_port", type=str, default="8080")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    run_viewer(model.extract(args), pipeline.extract(args), args.style_folder, viewer_port=args.viewer_port)





# print("final vgg features shape:", gaussian.final_vgg_features.shape)
# print("xyz shape:", gaussian._xyz.shape)
# print("cls shape:", gaussian._cls.shape)
# print("features_dc shape:", gaussian._features_dc.shape)
# print("features_rest shape:", gaussian._features_rest.shape)
# print("scaling shape:", gaussian._scaling.shape)
# print("rotation shape:", gaussian._rotation.shape)
# print("opacity shape:", gaussian._opacity.shape)
# print("max_radii2D shape:", gaussian.max_radii2D.shape)
# print("xyz_gradient_accum shape:", gaussian.xyz_gradient_accum.shape)
# print("denom shape:", gaussian.denom.shape)
# self._xyz = torch.empty(0)
# self._features_dc = torch.empty(0)
# self._features_rest = torch.empty(0)
# self._scaling = torch.empty(0)
# self._rotation = torch.empty(0)
# self._opacity = torch.empty(0)
# self.max_radii2D = torch.empty(0)
# self.xyz_gradient_accum = torch.empty(0)
# self.denom = torch.empty(0)
# self.optimizer = None
# self.percent_dense = 0
# self.spatial_lr_scale = 0
# self.setup_functions()
# self._cls = torch.empty(0)
# raise NotImplementedError("你的生命已如风中残烛！")


# show distribution of cls
# hist = torch.histc(gaussians._cls, bins=1000, min=0, max=999)
# frequency_dict = {i: int(hist[i]) for i in range(1000)}
# for i in range(1000):
#     if frequency_dict[i] < 100:
#         del frequency_dict[i]
# print("Frequency dict: ")
# for key, value in frequency_dict.items():
#     print(f"{key}: {value}")
# raise NotImplementedError("你的生命已如风中残烛！")


# def update_seg(gaussian, seg):
#     ori_xyz = gaussian._xyz.cpu().numpy()
#     seg_xyz = seg["xyz"].cpu().numpy()
#     seg_cls = seg["cls"]
#     neigh = NearestNeighbors(n_neighbors=1)
#     neigh.fit(seg_xyz)
#     _, indices = neigh.kneighbors(ori_xyz)
#     indices = indices.flatten()
#     ori_cls = seg_cls[indices]
#     gaussian._cls = torch.tensor(ori_cls, dtype=torch.long, device="cuda")
    
    # print("Update seg cls done!")


# seg_fet_dc = seg["fet_dc"]
# seg_fet_rest = seg["fet_rest"]
# near_fet_dc = seg_fet_dc[indices]

# ori_fet_dc = seg_fet_dc[indices]
# ori_fet_rest = seg_fet_rest[indices]
# gaussian._features_dc = ori_fet_dc
# gaussian._features_rest = ori_fet_rest

# # print xyz range
# xyz_min = gaussians._xyz.min(0)
# xyz_max = gaussians._xyz.max(0)
# print(f"xyz min: {xyz_min}, xyz max: {xyz_max}")


# show distribution of cls
# hist = torch.histc(gaussians._cls, bins=1000, min=0, max=999)
# frequency_dict = {i: int(hist[i]) for i in range(1000)}
# for i in range(1000):
#     if frequency_dict[i] < 100:
#         del frequency_dict[i]
# print("Frequency dict: ")
# for key, value in frequency_dict.items():
#     print(f"{key}: {value}")
# raise NotImplementedError("你的生命已如风中残烛！")


# calculate gaussians xyz range
# xyz shape: torch.Size([5834784, 3])
# xyz = seg_gaussians._xyz
# xyz_min = xyz.min(0)
# xyz_max = xyz.max(0)
# print(f"xyz min: {xyz_min}, xyz max: {xyz_max}")
# raise NotImplementedError("你的生命已如风中残烛！")
# [-44.7908, -34.0034, -24.1227]
# [57.4729,  9.3218, 35.2362]
# self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)


# xyz = gaussians._xyz
# xyz_min = xyz.min(0)
# xyz_max = xyz.max(0)
# print(f"xyz min: {xyz_min}, xyz max: {xyz_max}")
# raise NotImplementedError("你的生命已如风中残烛！")
# # [-46.2879, -53.1456, -24.5149]
# # [55.9820, 70.3039, 58.6648]


# Scene:
# Activate_sh_degree: 0
# Max_sh_degree: 3
# xyz shape: torch.Size([316266, 3])
# features_dc shape: torch.Size([0])
# features_rest shape: torch.Size([0])
# scaling shape: torch.Size([316266, 3])
# rotation shape: torch.Size([316266, 4])
# opacity shape: torch.Size([316266, 1])
# max_radii2D shape: torch.Size([0])
# xyz_gradient_accum shape: torch.Size([0])
# denom shape: torch.Size([0])
# percent_dense: 0
# spatial_lr_scale: 0

# Seg scene:
# Activate_sh_degree: 3
# Max_sh_degree: 3
# xyz shape: torch.Size([5834784, 3])
# mask shape: torch.Size([5834784])
# features_dc shape: torch.Size([5834784, 1, 3])
# features_rest shape: torch.Size([5834784, 15, 3])
# scaling shape: torch.Size([5834784, 3])
# rotation shape: torch.Size([5834784, 4])
# opacity shape: torch.Size([5834784, 1])
# max_radii2D shape: torch.Size([0])
# xyz_gradient_accum shape: torch.Size([0])
# denom shape: torch.Size([0])
# percent_dense: 0
# spatial_lr_scale: 0

# Seg feature:
# Activate_sh_degree: 0
# Max_sh_degree: 0
# xyz shape: torch.Size([5834784, 3])
# mask shape: torch.Size([5834784])
# scaling shape: torch.Size([5834784, 3])
# rotation shape: torch.Size([5834784, 4])
# opacity shape: torch.Size([5834784, 1])
# max_radii2D shape: torch.Size([0])
# xyz_gradient_accum shape: torch.Size([0])
# denom shape: torch.Size([0])
# percent_dense: 0
# spatial_lr_scale: 0

# 0: 1754
# 2: 833
# 6: 274
# 8: 124
# 9: 516
# 10: 135
# 11: 271
# 12: 125
# 14: 121
# 15: 487
# 17: 158
# 19: 578
# 20: 486
# 21: 270
# 22: 5507
# 23: 22738
# 24: 415
# 25: 1155
# 26: 424
# 27: 369
# 28: 642
# 29: 515
# 30: 500
# 31: 256
# 33: 151
# 34: 506
# 35: 336
# 36: 198
# 37: 441
# 38: 104
# 39: 1448
# 40: 297
# 41: 1150
# 42: 1725
# 43: 108
# 44: 341
# 45: 473
# 46: 1352
# 47: 623
# 48: 555
# 49: 1200
# 50: 704
# 51: 724
# 52: 1243
# 53: 3368
# 54: 1814
# 55: 1173
# 56: 1448
# 58: 625
# 59: 453
# 60: 340
# 61: 562
# 62: 3884
# 63: 330
# 64: 533
# 65: 5102
# 66: 2601
# 67: 1749
# 68: 3207
# 69: 724
# 70: 1595
# 71: 458
# 72: 925
# 73: 252
# 74: 549
# 75: 420
# 76: 207
# 78: 438
# 79: 7030
# 80: 410
# 81: 982
# 82: 325
# 83: 540
# 84: 3713
# 85: 27469
# 86: 566
# 87: 9103
# 88: 58577
# 89: 396
# 90: 896
# 91: 6073
# 92: 1224
# 93: 28768
# 94: 3000
# 95: 16266
# 96: 8552
# 97: 56704


# 1: 195, 
# 2: 622, 
# 3: 451, 
# 5: 216, 
# 6: 139, 
# 7: 806, 
# 10: 121, 
# 11: 342, 
# 12: 259, 
# 13: 255, 
# 15: 138, 
# 17: 153, 
# 18: 263, 
# 19: 143, 
# 23: 166, 
# 25: 372, 
# 29: 294, 
# 30: 135, 
# 31: 273, 
# 34: 473, 
# 37: 214, 
# 38: 185, 
# 39: 305, 
# 40: 226, 
# 41: 1015, 
# 42: 114, 
# 43: 555, 
# 45: 17118, 
# 46: 3985, 
# 47: 536, 
# 48: 773, 
# 49: 167, 
# 51: 101, 
# 55: 442, 
# 56: 234, 
# 57: 714, 
# 58: 260, 
# 59: 329, 
# 60: 722, 
# 61: 319, 
# 62: 643, 
# 63: 746, 
# 64: 4852, 
# 65: 322, 
# 66: 452, 
# 67: 1003, 
# 68: 108, 
# 71: 235, 
# 72: 171, 
# 73: 602, 
# 74: 1674, 
# 75: 690, 
# 77: 163, 
# 79: 779, 
# 80: 118, 
# 84: 302, 
# 85: 268, 
# 86: 1139, 
# 87: 331, 
# 88: 409, 
# 89: 143, 
# 90: 1137, 
# 91: 623, 
# 93: 465, 
# 94: 668, 
# 95: 177, 
# 96: 1893, 
# 97: 682, 
# 98: 116, 
# 99: 1143, 
# 100: 1079, 
# 101: 916, 
# 103: 154, 
# 104: 797, 
# 105: 252, 
# 106: 127, 
# 108: 147, 
# 111: 147, 
# 112: 329, 
# 113: 221, 
# 114: 25878, 
# 115: 1193, 
# 116: 525, 
# 117: 342, 
# 118: 626, 
# 119: 432, 
# 120: 455, 
# 121: 910, 
# 122: 3692, 
# 123: 151, 
# 124: 116, 
# 125: 2316, 
# 126: 1038, 
# 128: 163, 
# 132: 663, 
# 133: 2281, 
# 134: 1108, 
# 135: 183, 
# 136: 1852, 
# 137: 1821, 
# 138: 794, 
# 139: 1283, 
# 141: 735, 
# 142: 1729, 
# 143: 3000, 
# 144: 442, 
# 146: 20499, 
# 147: 1637, 
# 148: 122, 
# 150: 152, 
# 151: 260, 
# 153: 336, 
# 155: 491, 
# 157: 7336, 
# 159: 194, 
# 161: 585, 
# 162: 109, 
# 163: 350, 
# 164: 8716, 
# 166: 222, 
# 167: 3543, 
# 170: 194, 
# 171: 1156, 
# 173: 158, 
# 174: 138, 
# 175: 5225, 
# 176: 723, 
# 177: 4086, 
# 178: 393, 
# 180: 358, 
# 181: 263, 
# 182: 900, 
# 183: 295, 
# 184: 384, 
# 185: 1615, 
# 186: 974, 
# 187: 52154, 
# 188: 679, 
# 189: 662, 
# 190: 1207, 
# 193: 259, 
# 195: 3339, 
# 196: 301, 
# 198: 109, 
# 204: 15225, 
# 205: 50423, 
# 206: 3913, 
# 207: 10498