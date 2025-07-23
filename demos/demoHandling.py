import os
from pathlib import Path
import numpy as np
from agentlace.data.data_store import QueuedDataStore

class DemoHandling:
    """
    Encapsulates loading of Gymnasium-style demo .npz files
    into a DataStore (e.g., QueuedDataStore) for use with HER/RL agents.
    """
    def __init__(
        self,
        data_store: QueuedDataStore,
        demo_dir: str = '/data/data/serl/demos'
    ):
        """
        Parameters
        ----------
        data_store : QueuedDataStore
            The replay buffer or datastore to insert transitions into.
        demo_dir : str
            Directory where demo .npz files live by default.
        """
        self.data_store = data_store
        self.demo_dir = demo_dir

    @staticmethod
    def load_demos_to_her_buffer_gymnasium(
        data_store,
        demo_npz_path: str,
        combine_done: bool = True
    ):
        """
        Load a raw Gymnasium-style .npz of expert episodes into data_store.

        demo_npz_path must contain arrays named 'obs', 'acs', 'rewards',
        'terminateds', 'truncateds', 'info', and optionally 'dones'.

        If combine_done=True, done = terminated OR truncated OR done_buffer.
        """
        data = np.load(demo_npz_path, allow_pickle=True)

        obs_buffer   = data['obs']         # shape (N, T+1, ...)
        act_buffer   = data['acs']         # shape (N, T,   ...)
        rew_buffer   = data['rewards']     # shape (N, T)
        term_buffer  = data['terminateds'] # shape (N, T)
        trunc_buffer = data['truncateds']  # shape (N, T)
        info_buffer  = data['info']        # shape (N, T)
        done_buffer  = data.get('dones', term_buffer | trunc_buffer)

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

                raw_info = ep_info[t]
                if isinstance(raw_info, str):
                    import ast
                    info_t = ast.literal_eval(raw_info)
                else:
                    info_t = raw_info.copy()
                info_t["TimeLimit.truncated"] = bool(ep_trunc[t])

                data_store.insert(
                    dict(
                        observations=obs_t,
                        actions=a_t,
                        next_observations=next_obs_t,
                        rewards=r_t,
                        masks=1.0 - done_t,
                        dones=done_t
                    )
                )

        print(f"Loaded {num_demos} episodes from '{demo_npz_path}' "
              f"(combine_done={combine_done}).")

    @staticmethod
    def get_demo_path(relative_path: str) -> str:
        """
        Resolve a path relative to this file's location.
        """
        script_dir = Path(__file__).resolve().parent
        return str((script_dir / relative_path).resolve())


    def run(self, default_file: str = 'data_franka_reach_random_20.npz'):
        """
        Prompt for a file name (with a sensible default), then load it.
        """
        prompt = f"Please input the name of the file to load [{default_file}]: "
        
        file_name = input(prompt).strip() or default_file
        demo_file = os.path.join(self.demo_dir, file_name)

        self.load_demos_to_her_buffer_gymnasium(
            self.data_store,
            demo_file,
            combine_done=True # dones are truncated or terminated in this case.
        )


# if __name__ == "__main__":
#     # create your datastore; here we use a QueuedDataStore with capacity 2000
#     ds = QueuedDataStore(2000)
#     handler = DemoHandling(ds)
#     handler.run()
