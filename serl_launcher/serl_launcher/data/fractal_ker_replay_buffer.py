import gym
import numpy as np
from serl_launcher.data.dataset import DatasetDict
from serl_launcher.data.replay_buffer import ReplayBuffer
from transforms3d.euler import euler2mat
import copy


class FractalKerReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        branch_method: str,
        split_method: str,
        workspace_width: int,
        n_KER : int,
        max_z_theta: float,
        kwargs: dict
    ):
        self.current_branch_count=1
        self.update_max_traj_length = False
        self.n_KER = n_KER # Number of reflectional planes to generate. Number of new traj = n_ker - 1
        self.max_z_theta = max_z_theta # Max theta possible for generating reflectional planes
        
        self.env_type = 'simulation'
        self.pos_idx = np.array([0,1,2])
        self.vel_idx = np.array([4,5,6])
        self.bp_idx = np.array([7,8,9]) # Block position
        self.acts_idx = np.array([0,1,2])
        self.translational_obs = np.array([self.pos_idx, self.vel_idx, self.bp_idx])
        self.rotational_obs = np.array([])
        self.action_space_dim = action_space.shape[0]

        if "z_theta_list" in kwargs.keys():
            self.z_theta_list = kwargs["z_theta_list"]
        else:
            self.z_theta_list = []

        method_check = "split_method"
        match split_method:
            case "time":
                assert "max_depth" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_depth")
                self.max_depth=kwargs["max_depth"]

                assert "max_traj_length" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "max_traj_length")
                self.max_traj_length=kwargs["max_traj_length"]

                assert "alpha" in kwargs.keys(), self._handle_bad_args_(method_check, split_method, "alpha")
                self.alpha=kwargs["alpha"]
                update_max_traj_length = True
                self.split = self.time_split 

            case "constant":
                self.split = self.constant_split

            case "test":
                self.split = self.test_split
            case _:
                raise ValueError("incorrect value passed to split_method")
        
        method_check = "branch_method"
        match branch_method:
            case "fractal":
                assert "max_depth" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "max_depth")
                self.max_depth=kwargs["max_depth"]

                assert "branching_factor" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branching_factor")
                self.branching_factor=kwargs["branching_factor"]

                assert "start_num" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "start_num")
                self.start_num=kwargs["start_num"]

                self.branch = self.fractal_branch
            case "contraction":
                assert "start_num" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "start_num")
                self.start_num=kwargs["start_num"]

                # Add branching_factor for contraction method
                assert "branching_factor" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branching_factor")
                self.branching_factor = kwargs["branching_factor"]

                self.branch = self.fractal_contraction
            case "linear":
                raise NotImplementedError("linear branch method is not yet implemented")
                # TODO: Implement constant branch method
                # assert "branch_count_rate_of_change" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "branch_count_rate_of_change")
                # self.branch_count_rate_of_change=kwargs["branch_count_rate_of_change"]
                # del kwargs["branch_count_rate_of_change"]

                # assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_(method_check, branch_method, "starting_branch_count")
                # self.current_branch_count = kwargs["starting_branch_count"]
                # del kwargs["starting_branch_count"]

                # self.branch = self.linear_branch
            case "constant":
                assert "starting_branch_count" in kwargs.keys(), self._handle_bad_args_("branch_method", branch_method, "starting_branch_count")
                self.current_branch_count = kwargs["starting_branch_count"]
                del kwargs["starting_branch_count"]

                self.branch = self.constant_branch
            case "test":
                self.branch = self.test_branch
            case _:
                raise ValueError("incorrect value passed to branch_method")
        
        self.x_obs_idx = kwargs["x_obs_idx"]
        self.y_obs_idx = kwargs["y_obs_idx"]
        self.workspace_width = workspace_width
        self.timestep = 0
        self.current_depth = 0
        self.branch_index = [workspace_width/2]
    


        self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
        constant = self.workspace_width/(2 * self.current_branch_count)
        for i in range(0, self.current_branch_count):
            self.branch_index[i] = (2 * i + 1) * constant

        for k in kwargs.keys():
            print(f"\033[33mWARNING \033[0m argument \"{k}\" not used")
        
        # TODO
        #   Create more methods with tests
        # 
        # Future considerations:
        #   when dealing with x and y for transforms, consider two versions:
        #       radial trees at different angles can have strong fractal symmetry built in but will be more difficult to evenly space lowest level transforms
        #       grid-based will inherently evenly space lowest transforms, but fractal symmetry will be restricted to 9 branching_factor (original plus 8) in order to conform

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

    def linear_vector_symmetric_with_rot_plane(self, data_dict, rot_z_theta, inv_rot_z_theta):
        # Point 'a' position = v_l_a
        for obs in self.translational_obs:
            o_data = data_dict["observations"][obs]
            o_data_hat = np.dot(inv_rot_z_theta,o_data)
            o_data_hat[1] = -o_data_hat[1]
            s_data =  np.dot(rot_z_theta,o_data_hat)
            data_dict["observations"][obs] = s_data

            o_data = data_dict["next_observations"][obs]
            o_data_hat = np.dot(inv_rot_z_theta,o_data)
            o_data_hat[1] = -o_data_hat[1]
            s_data =  np.dot(rot_z_theta,o_data_hat)
            data_dict["next_observations"][obs] = s_data
            
        # TODO fix spaghetti
        o_data = data_dict["actions"][self.acts_idx]
        o_data_hat = np.dot(inv_rot_z_theta,o_data)
        o_data_hat[1] = -o_data_hat[1]
        s_data =  np.dot(rot_z_theta,o_data_hat)
        data_dict["actions"][self.acts_idx] = s_data
        return

        
    
    # def reflect_orientation_with_rot_plane(self, o_data, rot_z_theta, inv_rot_z_theta):
    #     # reflects orientation about plane given by theta
    #     o_mat = euler2mat(o_data[0],o_data[1],o_data[2], axes='rxyz')
    #     o_mat_hat = np.matmul(o_mat, inv_rot_z_theta)
    #     o_euler = np.array(mat2euler(o_mat_hat, axes='rxyz'), dtype=np.float32)
    #     o_euler[0] = -o_euler[0]
    #     o_euler[2] = -o_euler[2]
    #     s_mat = euler2mat(o_euler[0],o_euler[1],o_euler[2], axes='rxyz')
    #     s_mat_hat = np.matmul(s_mat, rot_z_theta)
    #     s_euler = np.array(mat2euler(s_mat_hat, axes='rxyz'),dtype=np.float32)
    #     return s_euler.copy()


    def ker_insert(self, base_data_dict):
        ''' Reformats transformed episodes and inserts them into the replay buffer
        '''
        data_dict = copy.deepcopy(base_data_dict)

        # generate angles for reflection
        if not self.z_theta_list:
            self.z_theta_list = []
            for _ in range(self.n_KER - 1):
                self.z_theta_list.append(np.random.uniform(0, self.max_z_theta))

        y_axis_flip = euler2mat(0, 0, np.pi, axes="rxyz")

        # insert original + reflected transitions
        super().insert(data_dict)
        new_data_dict = copy.deepcopy(data_dict)

        self.linear_vector_symmetric_with_rot_plane(new_data_dict, y_axis_flip, y_axis_flip)
        # self.reflect_orientation_with_rot_plane()

        super().insert(new_data_dict)

        for z_theta in self.z_theta_list:
            # initial transitions
            rot_z_theta = euler2mat(0, 0, z_theta, axes='rxyz')
            inv_rot_z_theta = euler2mat(0, 0, -z_theta, axes='rxyz')
            new_data_dict = copy.deepcopy(data_dict)
            self.linear_vector_symmetric_with_rot_plane(new_data_dict, rot_z_theta, inv_rot_z_theta)
            # self.reflect_orientation_with_rot_plane()

            super().insert(new_data_dict)

            # flipped across y-axis
            self.linear_vector_symmetric_with_rot_plane(new_data_dict, y_axis_flip, y_axis_flip)
            # self.reflect_orientation_with_rot_plane()

            super().insert(new_data_dict)

    def _handle_bad_args_(self, type: str, method: str, arg: str) :
        return f"\033[31mERROR: \033[0m{arg} must be defined for {type} \"{method}\""
    

    def transform(self, data_dict: DatasetDict, transform: np.array):
        #   return data_dict with positional arguments += translation
        data_dict["observations"] += transform
        data_dict["next_observations"] += transform
        
    # FOR TESTING ONLY
    def test_branch(self):
        self.current_depth += 1
        return 1
    
    def fractal_branch(self):
        # return a new number of branches = branching_factor ^ depth
        return self.branching_factor ** self.current_depth
    
    def fractal_contraction(self):
        # return 
        b = self.branching_factor  
        d = self.current_depth
        top_b = self.start_num

        return int( top_b/( b**(d-1) ))
    
    # REQUIRES TESTING
    def constant_branch(self):
        # return current number of branches
        return self.current_branch_count
    
    # REQUIRES TESTING
    # def linear_branch(self):
    #     # return a new number of branches = branches_count + n
    #     return self.current_branch_count + self.branch_count_rate_of_change
    
    # FOR TESTING ONLY
    def test_split(self, data_dict: DatasetDict):
        return True
            
    # REQUIRES TESTING
    def time_split(self, data_dict: DatasetDict):
        if self.timestep % (self.max_traj_length//self.max_depth) or self.current_depth >= self.max_depth:
            return False
        self.current_depth += 1
        return True
    
    def constant_split(self, data_dict: DatasetDict):
        return True
    
    def insert(self, data_dict_not: DatasetDict):
        
        data_dict = copy.deepcopy(data_dict_not)

        # Update number of branches if needed
        if self.split(data_dict):
            temp = self.current_branch_count
            self.current_branch_count = self.branch()
            if temp != self.current_branch_count:
                self.branch_index = np.empty(self.current_branch_count, dtype=np.float32)
                constant = self.workspace_width/(2 * self.current_branch_count)
                for i in range(0, self.current_branch_count):
                    self.branch_index[i] = (2 * i + 1) * constant
        

        # Initialize to extreme x and y
        x = -self.workspace_width/2
        transform = np.zeros_like(data_dict["observations"])
        transform[self.x_obs_idx] = x
        transform[self.y_obs_idx] = x
        self.transform(data_dict, transform)

        # Transform and insert transitions (multiprocessing in the future)
        for x in range(0, self.current_branch_count):
            transform[self.x_obs_idx] = self.branch_index[x]
            for y in range(0, self.current_branch_count):
                new_data_dict = copy.deepcopy(data_dict)
                transform[self.y_obs_idx] = self.branch_index[y]
                self.transform(new_data_dict, transform)
                self.ker_insert(new_data_dict)


        self.timestep += 1
        if data_dict["dones"]:
            self.current_depth = 0
            if self.update_max_traj_length:
                self.max_traj_length = int(self.timestep * self.alpha + self.max_traj_length * (1 - self.alpha))
            self.timestep = 0

        