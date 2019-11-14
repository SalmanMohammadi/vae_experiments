mnist_vae_experiment_1
---
config = Config(
	# dataset
	dataset={
		digits=[0, 1],
		n_obs=2
	},
	# transformations
	transformations={
		trans_per_image=100,
		max_angle=45,
		max_brightness=0.,
		max_noise=0.,
		max_scale=0.
	},
	#model
	model={
		epochs=100,
		lr=0.01,
		z_size=2
	}
)
---
no_experiments = 
variables = {}
---