import os
from pathlib import Path
import numpy as np
from agentlace.data.data_store import QueuedDataStore

class DemoHandling:
    """
    Koads an .npz file containing demonstration data into a data object.
    This class is designed to work with Gymnasium-style demonstration data
    and is intended to be used with a QueuedDataStore or similar data store.

    The .npz file should contain the following arrays:
      - 'obs'            : shape (N, T+1, *obs_shape*), list of observations
      - 'acs'            : shape (N, T, *act_shape*),   list of actions
      - 'rewards'        : shape (N, T),                list of rewards
      - 'terminateds'    : shape (N, T),                list of terminated flags
      - 'truncateds'     : shape (N, T),                list of truncated flags
      - 'info'          : shape (N,  T),                list of info dicts
      - 'dones'         : shape (N,  T),                list of done flags (if available)

    Parameters
    ----------
    demo_dir : str
        Directory where demo .npz files live by default.
    file_name : str
        Name of the demo file to load. If not provided, a default will be used.
    """
    def __init__(
        self,
        demo_dir: str = '/data/data/serl/demos',
        file_name: str = 'data_franka_reach_random_20.npz'
    ):

        self.debug = False  # Set to True for debugging purposes
        self.demo_dir = demo_dir
        self.transition_ctr = 0  # Global counter for transitions across all episodes

        # Load the demo data from the .npz file 

        # Check if the demo directory exists
        if not os.path.exists(self.demo_dir):
            raise FileNotFoundError(f"Demo directory '{self.demo_dir}' does not exist.")
        
        # Construct the full path to the demo file
        self.demo_npz_path = os.path.join(self.demo_dir, file_name)
        if not os.path.isfile(self.demo_npz_path):
            raise FileNotFoundError(f"Demo file '{self.demo_npz_path}' does not exist.")
        
        # Load the .npz file
        self.data = np.load(self.demo_npz_path, allow_pickle=True)

    def get_num_transitions(self):
        """
        Returns the total number of transitions counted in the demo data.
        """
        return self.data["transition_ctr"] if "transition_ctr" in self.data else 0

    def insert_data_to_buffer(self,data_store: QueuedDataStore): 
        """
        Load a raw Gymnasium-style .npz of expert episodes into data_store.
        The .npz file must contain arrays named 'obs', 'acs', 'rewards',
        'terminateds', 'truncateds', 'info', and optionally 'dones'.
        """
        
        obs_buffer   = self.data['obs']         # shape (N, T+1, ...)
        act_buffer   = self.data['acs']         # shape (N, T,   ...)
        rew_buffer   = self.data['rewards']     # shape (N, T)
        term_buffer  = self.data['terminateds'] # shape (N, T)
        trunc_buffer = self.data['truncateds']  # shape (N, T)
        info_buffer  = self.data['info']        # shape (N, T)
        done_buffer  = self.data.get('dones', term_buffer | trunc_buffer)

        num_demos = obs_buffer.shape[0]

        for ep in range(num_demos):
            ep_obs   = obs_buffer[ep]
            ep_acts  = act_buffer[ep]
            ep_rews  = rew_buffer[ep]
            ep_terms = term_buffer[ep]
            ep_trunc = trunc_buffer[ep]
            ep_done  = done_buffer[ep]
            ep_info  = info_buffer[ep]

            T = len(ep_acts)
            for t in range(T):
                obs_t       = ep_obs[t]
                next_obs_t  = ep_obs[t+1]
                a_t         = ep_acts[t]
                r_t         = float(ep_rews[t])
                done_t      = bool(ep_done[t] or ep_terms[t] or ep_trunc[t])
                # masks will be created right before insert below

                if self.debug:
                    print(f"Demo {ep}, Step {t}: Obs={obs_t}, Action={a_t}, Reward={r_t}, Done={done_t}")

                data_store.insert(
                    dict(
                        observations     =obs_t,
                        actions          =a_t,
                        next_observations=next_obs_t,
                        rewards          =r_t,
                        masks            =1.0 - done_t,
                        dones            =done_t
                    )
                )

        print(f"Loaded {num_demos} episodes from '{self.demo_npz_path}' ")


# if __name__ == "__main__":
#     # create your datastore; here we use a QueuedDataStore with capacity 2000
#     ds = QueuedDataStore(2000)
#     handler = DemoHandling(ds,
#                            demo_dir='/data/data/serl/demos',
#                            file_name='data_franka_reach_random_20.npz')
    
#     # Idenitfy the total number of transitions in the datastore
#     print(f'We have {handler.data["transition_ctr"]} transitions in the datastore.')
    
#     # Load the demo data into the data_store
#     handler.insert_data_to_buffer()
