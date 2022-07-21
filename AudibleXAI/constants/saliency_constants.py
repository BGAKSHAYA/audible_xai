INNVESTIGATE_SALIENCY_METHODS = ["input", "random", "gradient", "gradient.baseline", "input_t_gradient", "deconvnet",
                                 "guided_backprop", "integrated_gradients", "smoothgrad", "lrp", "lrp.z", "lrp.z_IB",
                                 "lrp.epsilon", "lrp.epsilon_IB", "lrp.w_square", "lrp.flat", "lrp.alpha_beta",
                                 "lrp.alpha_2_beta_1", "lrp.alpha_2_beta_1_IB", "lrp.alpha_1_beta_0",
                                 "lrp.alpha_1_beta_0_IB",
                                 "lrp.z_plus", "lrp.z_plus_fast", "lrp.sequential_preset_a", "lrp.sequential_preset_b",
                                 "lrp.sequential_preset_a_flat", "lrp.sequential_preset_b_flat",
                                 "lrp.sequential_preset_b_flat_until_idx", "deep_taylor", "deep_taylor.bounded",
                                 "deep_lift.wrapper", "pattern.net", "pattern.attribution"]


CUSTOM_SALIENCY_METHODS = [
    "INPUT_TIMES_GRADIENT",
    "INTEGRATED_GRADIENTS",
    "GRAD_CAM",
    "OCCLUSION",
    "LIME"
]

# We are interested to investigate these saliency methods
SALIENCY_METHODS = [
    "gradient", "input_t_gradient", "INTEGRATED_GRADIENTS", "deconvnet",
    "deep_taylor", "lrp.sequential_preset_a",  # lrp epsilon rule for dense layers, lrp.alpha1beta0 rule for conv layers
    "GRAD_CAM", "OCCLUSION", "LIME"
]


saliency_name_mapper = {
	"gradient": "G",
	"input_t_gradient": "ITG",
	"INTEGRATED_GRADIENTS": "IG",
    "deconvnet": "DNET",
    "deep_taylor": "DTD",
    "lrp.sequential_preset_a": "LRP",
    "GRAD_CAM": "GCAM",
    "LIME": "LIME",
	"OCCLUSION": "Occlusion"
}