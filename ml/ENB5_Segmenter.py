import keras
import tensorflow as tf

class Decoder(keras.layers.Layer):
	def __init__(
		self,
		decoder_dims,
		encoder_blocks,
		dropout_ratio,
		num_classes,
		output_size,
		decoder_upscale_size,
		**kwargs
	):
		super().__init__(**kwargs)
		self.decoder_dims = decoder_dims
		self.output_size = output_size
		self.decoder_upscale_size = decoder_upscale_size

		self.mlps = []
		for i in range(encoder_blocks):
			self.mlps.append(
				keras.layers.Dense(decoder_dims, name=f'mlp_{i}')
			)

		self.linear_fuse = keras.layers.Conv2D(
            filters=decoder_dims, 
			kernel_size=1, 
			use_bias=False, 
			name="linear_fuse"
        )
		self.batch_norm = keras.layers.BatchNormalization(name="batch_norm")
		self.activation = keras.layers.Activation("relu")
		self.dropout = keras.layers.Dropout(dropout_ratio)
		self.classifier = keras.layers.Conv2D(
			filters=num_classes, 
			kernel_size=1,
			name="classifier"
		)
	
	def call(
		self,
		encoder_outputs,
	):
		mlp_outputs = []
		for (output, mlp) in zip(encoder_outputs, self.mlps):
			#flattening for dense layer
			height = output.shape[1]
			width = output.shape[2]
			channels = output.shape[3]
			flat_output = tf.reshape(output, (-1, height*width, channels))
			mlp_output = mlp(flat_output)
			
			#reshaping into batch, height, width, filters
			mlp_output = tf.reshape(mlp_output, (-1, height, width, self.decoder_dims))
			# upsample
			mlp_output = tf.image.resize(mlp_output, size=self.decoder_upscale_size, method="bilinear")
			mlp_outputs.append(mlp_output)

		hidden_states = self.linear_fuse(tf.concat(mlp_outputs, axis=-1))
		hidden_states = self.batch_norm(hidden_states)
		hidden_states = self.activation(hidden_states)
		hidden_states = self.dropout(hidden_states)
		logits = self.classifier(hidden_states)
		upsampled_logits = tf.image.resize(logits, size=self.output_size, method="bilinear")
		return upsampled_logits

class Encoder(keras.layers.Layer):
	def __init__(self, input_shape, **kwargs):
		super().__init__(**kwargs)
		base_model = tf.keras.applications.EfficientNetB5(
			include_top=False,
			weights="imagenet",
			input_shape=input_shape,
		)
		self.encoder_layers = [
            base_model.layers[41],
            base_model.layers[115],
            base_model.layers[189],
            base_model.layers[293],
            base_model.layers[396],
            base_model.layers[530],
            base_model.layers[573],
        ]
		encoder_outputs = [layer.output for layer in self.encoder_layers]
		self.encoder_model = keras.Model(base_model.input, encoder_outputs)
		del base_model

	def call(self, pixel_values):
		encoder_outputs = self.encoder_model(pixel_values)
		return encoder_outputs
	

class EfficientNetB5_Segmenter(keras.Model):
	def __init__(self, config, **kwargs):
		super().__init__(**kwargs)
		self.encoder = Encoder(config.input_shape)
		self.decoder = Decoder(
			config.decoder_dims,
			config.encoder_blocks,
			config.dropout_ratio,
			config.num_classes,
			config.output_size,
			config.decoder_upscale_size #for B5 = [256, 256]
		)
	def call(self, pixel_values):
		encoder_outputs = self.encoder(pixel_values)
		decoder_output = self.decoder(encoder_outputs)
		return decoder_output


class EfficientNet_Segmenter_Config():
	def __init__(
		self,
		input_shape = (512, 512, 3),
		decoder_dims = 256,
		encoder_blocks = 7,
		dropout_ratio = 0.1,
		num_classes = 2,
		output_size = [512, 512],
		decoder_upscale_size = [256, 256] #must be equal to largest output of encoder
	):
		self.input_shape = input_shape
		self.decoder_dims = decoder_dims
		self.encoder_blocks = encoder_blocks
		self.dropout_ratio = dropout_ratio
		self.num_classes = num_classes
		self.output_size = output_size
		self.decoder_upscale_size = decoder_upscale_size