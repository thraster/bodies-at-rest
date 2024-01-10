import pickle
import open3d as o3d
import utils
import numpy as np
import cv2
import scipy.io
import os

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f,encoding='latin1')
    


class load_real():
    '''
    加载一个指定的.p文件,

    .p文件具有的keys:
    1. markers_xyz_m:   人体模型的3d关节点坐标 (24,3)
    2. body_shape:      smpl模型的shape参数, 10个
    3. mesh_contact:    模型与床的接触图 (27,64)
    4. mesh_depth:      模型深度图 (27, 64)
    5. root_xyz_shift:  根节点偏移 (1,3)
    6. body_height:     身高(m)
    7. body_mass:       体重(kg)
    8. images:          可能是pressure image (1728*1 = 64*27)
    9. joint_angles:    smpl模型的pose参数 (24,3)
    10. bed_angle_deg:  床角度偏移？ 1位float
    '''

    def __init__(self, pfile_path):
        print(f"loading target: {pfile_path}")
        '''
        p_select.p: 选择过的.p
        participant_info_red.p 被实验人员的信息
        prescribed.p 没有被选择过的.p
        -----------------
        dict_keys(['pmat_corners', 'pose_type', 'depth', 'pc', 'RGB', 'images']),
        dict_keys(['height_in', 'gender', 'weight_lbs', 'prescribed_pose_type', 'p_select_pose_type']),
 
        '''
        self.pfile_path = pfile_path
        self.dat1 = load_pickle(self.pfile_path+'\\p_select.p') # p_select.p
        self.dat2 = load_pickle(self.pfile_path+'\\participant_info_red.p') # participant_info_red.p
        self.dat3 = load_pickle(self.pfile_path+'\\prescribed.p') # prescribed.p

        # ['pmat_corners', 'pose_type', 'depth', 'pc', 'RGB', 'images']
        # pmat_corners
        # depth: 深度图
        # pc: pressure contact
        # RGB: RGB图
        # images: pressure mat 64x27
        self.keys_data = self.dat1.keys()
        # ['height_in', 'gender', 'weight_lbs', 'prescribed_pose_type', 'p_select_pose_type']
        self.keys_info = self.dat2.keys()

        self.length1 = len(self.dat1['pmat_corners'])

    # def get_skeleton(self, iter=0, viz=False):
    #     '''
    #     visuilzing the 3D skeleton
    #     '''
    #     try:
    #         nodes_3d = self.dat['markers_xyz_m'][iter].reshape(24,3)
    #         print(nodes_3d.dtype)
    #         if viz == True:
    #             # 创建一个Open3D点云对象
    #             pcd = o3d.geometry.PointCloud()

    #             # 设置点云数据
    #             pcd.points = o3d.utility.Vector3dVector(nodes_3d)

    #             # 创建可视化窗口
    #             o3d.visualization.draw_geometries([pcd])
    #         return nodes_3d
        
    #     except:
    #         print(f"iter = {iter} out of boundary: 0~{len(self.dat['markers_xyz_m'])-1}")

    # def get_trans(self, iter = 0):
    #     """
    #     返回根节点偏移量
    #     """
    #     return self.dat['root_xyz_shift'][iter]

    # def get_mesh(self, iter=0, viz=False):
    #     '''
    #     visuilzing the 3D smpl mesh of human body
    #     '''
        
    #     try:
    #         shapes = self.dat['body_shape'][iter]
    #         poses = self.dat['joint_angles'][iter]

    #         if self.gender == 'm':
    #             smpl = pickle._Unpickler(open(r"smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl", "rb"), encoding='latin1')
    #             smpl = smpl.load()
    #         elif self.gender == 'f':
    #             smpl = pickle._Unpickler(open(r"smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl", "rb"), encoding='latin1')
    #             smpl = smpl.load()

    #         # 计算shape参数下对应的身材
    #         v_shaped = smpl['shapedirs'].dot(shapes) + smpl['v_template']

    #         # 计算 T-pose 下 joints 位置
    #         J = smpl['J_regressor'].dot(v_shaped)     

    #         # 计算受 pose 影响下调整臀部之后的 vertices
    #         v_posed = v_shaped + smpl['posedirs'].dot(utils.posemap(poses))   

    #         # 将 v_posed 变成齐次坐标矩阵 (6890, 4)
    #         v_posed_homo = np.vstack((v_posed.T, np.ones([1, v_posed.shape[0]])))   
            
    #         # 将关节点的轴角 (axial-angle) 形状为 [24, 3]
    #         poses = poses.reshape((-1,3))

    #         # 定义 SMPL 的关节树, 开始骨骼绑定操作
    #         id_to_col = {smpl['kintree_table'][1,i] : i for i in range(smpl['kintree_table'].shape[1])}
    #         parent = {i : id_to_col[smpl['kintree_table'][0,i]] for i in range(1, smpl['kintree_table'].shape[1])}

    #         rodrigues = lambda x: cv2.Rodrigues(x)[0]
    #         Ts = np.zeros([24,4,4])

    #         # 首先计算根结点 (0) 的相机坐标变换, 或者说是根结点相对相机坐标系的位姿
    #         T = np.zeros([4,4])
    #         T[:3, :3] = rodrigues(poses[0])     # 轴角转换到旋转矩阵，相对相机坐标而言
    #         T[:3, 3] = J[0]                     # 根结点在相机坐标系下的位置
    #         T[3, 3] = 1                         # 齐次矩阵，1
    #         Ts[0] = T

    #         # 计算子节点 (1~24) 的相机坐标变换
    #         for i in range(1,24):
    #             # 首先计算子节点相对父节点坐标系的位姿 [R|t]
    #             T = np.zeros([4,4])
    #             T[3, 3] = 1

    #             # 计算子节点相对父节点的旋转矩阵 R
    #             T[:3, :3] = rodrigues(poses[i])

    #             # 计算子节点相对父节点的偏移量 t
    #             T[:3, 3]  = J[i] - J[parent[i]]

    #             # 然后计算子节点相对相机坐标系的位姿
    #             Ts[i] = np.matmul(Ts[parent[i]], T) # 乘上其父节点的变换矩阵

    #         global_joints = Ts[:, :3, 3].copy() # 所有关节点在相机坐标系下的位置

    #         # 计算每个子节点相对 T-pose 时的位姿矩阵
    #         # 由于子节点在 T-pose 状态下坐标系朝向和相机坐标系相同，因此旋转矩阵不变, 只需要减去 T-pose 时的关节点位置就行

    #         for i in range(24):
    #             R = Ts[i][:3, :3]
    #             t = Ts[i][:3, 3] - R.dot(J[i])              # 子节点相对T-pose的偏移 t
    #             Ts[i][:3, 3] = t

    #         # 开始蒙皮操作，LBS 过程
    #         vertices_homo = np.matmul(smpl['weights'].dot(Ts.reshape([24,16])).reshape([-1,4,4]), v_posed_homo.T.reshape([-1, 4, 1]))
    #         vertices = vertices_homo.reshape([-1, 4])[:,:3]    # 由于是齐次矩阵，取前3列
    #         joints = smpl['J_regressor'].dot(vertices)     # 计算 pose 下 joints 位置，基本与 global_joints 一致

    #         # utils.render(vertices, smpl['f'], joints)
    #         if viz == True:
    #             utils.render(vertices, smpl['f'], global_joints)

    #         return vertices

    #     except:
    #         print(f"iter = {iter} out of boundary: 0~{len(self.dat['markers_xyz_m'])-1}")

    def get_pmap(self, iter=0, viz=False):
        '''
        visuilzing the pressure map of a given iter
        '''
        try:
            # 64*27的图片
            img = self.dat1['images'][iter]
            print(img.dtype)
            # 使用 cv2.imshow() 显示图像
            if viz == True:
                cv2.imshow(f'pressure map of iter: {iter}', img)

                # 等待按键响应并关闭窗口
                cv2.waitKey(0)

                # 销毁所有窗口
                cv2.destroyAllWindows()
            return img
        
        except:
            print(f"iter = {iter} out of boundary: 0~{len(self.dat['markers_xyz_m'])-1}")


    # def to_file_mod1(self, save_path):
    #     '''
    #     提取需要的pressure map和skeleton, 分别保存为.png和.mat
    #     save_path------train\test---name--- name_001.png, name_002.png, ...name_001.mat, name_002.mat, ...
    #     '''
    #     print(f'coverting file {self.name} to pressure map and skeleton...')
    #     root_dir = os.path.join(save_path)
    #     for i in range(self.length):
    #         print(f'frame {i}', end='\r')
    #         if "test_" in self.name:
    #             temp_dir = os.path.join(root_dir, 'test')
    #         elif "train_" in self.name:
    #             temp_dir = os.path.join(root_dir, 'train')

    #         dir = os.path.join(temp_dir, self.name[:-2])
    #         os.makedirs(dir, exist_ok=True)

    #         pressure_map_path = os.path.join(dir, f'{i}.png')
    #         skeleton_annotations_path = os.path.join(dir,  f'{i}.mat')
    #         cv2.imwrite(pressure_map_path, self.get_pmap(i))
    #         scipy.io.savemat(skeleton_annotations_path, {'3D_skeleton_annotation':self.get_skeleton(i),
    #                                                      'trans':self.get_trans(i)})



if __name__ == "__main__":
    # import os
    data = load_real(r"D:\workspace\python_ws\bodies-at-rest-master\data_BR\real\S103")
    data.get_pmap(1,True)



    # root_path = r'D:\workspace\python_ws\bodies-at-rest-master\data_BR\synth'


    # for root, dirnames, filenames in os.walk(root_path):
    #     # print(root)
    #     # print(filenames)
    #     # print(dirnames)
    #     for filename in filenames:
    #         sbj = load_real(root+ '\\' +filename)
    #         # sbj.to_file_mod1(r"D:\workspace\python_ws\pose-master\dataset")
    #         img = sbj.get_pmap()
    #         skel = sbj.get_skeleton()
    #         print(img.dtype)
    #         print(skel.dtype)
    #         ''''''
    #         break
    #     break
    # # sbj = load_synth(r"data_BR\synth\crossed_legs\test_roll0_xl_f_lay_set1both_500.p")
    # # sbj.to_file_mod1(r"D:\workspace\python_ws\bodies-at-rest-master\dataset")

