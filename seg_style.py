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

        # Add partially transfer arguments
        self.scene_index = 0
        self.index_to_cls = {
            0: -1, 
            1: 21, 
            2: 20, 
            3: 24, 
            4: 23, 
            5: 17, 
            6: 10, 
            7: 27, 
            8: 28, 
            9: 12, 
        }
        self.index_to_name = {
            0: "All",
            1: "Table",
            2: "Flower",
            3: "Floor",
            4: "House",
            5: "Tree",
            6: "Wall",
            7: "Grass near tree",
            8: "Grass aroud table",
            9: "Grass near flower pot",
        }

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


        def create_click_handler(index):
            def on_click(_):
                self.need_update = True
                self.scene_index = index
            return on_click

        with self.server.add_gui_folder("Style Transfer"):
            self.style_img_path_text = self.server.add_gui_text(
                    "Style Image",
                    initial_value="",
                    hint="Path to style image",                
                )
            
            self.display_style_img = self.server.add_gui_checkbox("Display Style Image", initial_value=True)

            if wikiart_img_paths is not None:
                self.random_style_button = self.server.add_gui_button("Random Style")

            with self.server.add_gui_folder("Partially Transfer"):
                for i in range(10):
                    # button = self.server.add_gui_button(f"Part {i}")
                    button = self.server.add_gui_button(self.index_to_name[i])
                    button.on_click(create_click_handler(i))

        
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

                rendering = render(view, self.gaussians, self.pipeline, self.background, override_color=self.override_color, index=self.index_to_cls[self.scene_index])["render"]
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