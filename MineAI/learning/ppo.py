import torch.optim as optim

from MineAI.memory.trajectory import PPOTrajectory


class PPO:
    ''' Inspired by https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py '''
    def __init__(self, env, actor, critic, epochs=50, steps_per_epoch=4000, discount_factor=0.99, gae_discount_factor=0.97, clip_ratio=0.2, target_kl=0.01, actor_lr=3e-4,
                 critic_lr=1e-3, train_actor_iters=80, train_critic_iters=80, save_freq=10):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Environment for the agent to interact with; already initialized
        actor : torch.nn.Module
            Neural network to use as the policy; already initialized
        critic : torch.nn.Module
            Neural network to use as the value function; already initialized
        epochs : int, optional
            Number of policy updates to perform after sampling experience
        steps_per_epoch : int, optional
            Number of steps of interaction with the environment per epoch
        discount_factor : float, optional
            Used to weight preference for long-term reward (aka gamma)
        gae_discount_factor : float, optional
            Used to weight preference for long-term advantage (aka lambda)
        clip_ratio : float, optional
            Maximum allowed divergence of the new policy from the old policy in the objective function (aka epsilon)
        target_kl : float, optional
            Target KL divergence for policy updates; used in model selection (early stopping)
        actor_lr : float, optional
            Learning rate for the actor module
        critic_lr : float, optional
            Learning rate for the critic module
        train_actor_iters : int, optional
            Number of iterations to train the actor per epoch
        train_critic_iters : int, optional
            Number of iterations to train the critic per epoch
        save_freq : int, optional
            Rate in terms of number of epochs that the actor and critic models are saved to disk
        """
        # Environment & Agent
        self.env = env
        self.actor = actor
        self.critic = critic

        # Training duration
        self.epochs = epochs,
        self.steps_per_epoch = steps_per_epoch
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters
        self.save_freq = save_freq

        # Learning hyperparameters
        self.discount_factor = discount_factor
        self.gae_discount_factor = gae_discount_factor
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def run(self):
        ''' Runs the proximal policy optimization algorithm '''


        for e in range(self.epochs):
            trajectory_buffer = PPOTrajectory(max_buffer_size=self.steps_per_epoch, discount_factor=self.dicsount_factor, gae_discount_factor=self.gae_discount_factor)
            actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

            observation = self.env.reset()
            t_return = 0.0
            for t in range(self.steps_per_epoch):
                
                

