??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
?
embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameembedding_6/embeddings
?
*embedding_6/embeddings/Read/ReadVariableOpReadVariableOpembedding_6/embeddings*
_output_shapes
:	?d*
dtype0
?
embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*'
shared_nameembedding_7/embeddings
?
*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes
:	?d*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
??*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	?*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/embedding_6/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*.
shared_nameAdam/embedding_6/embeddings/m
?
1Adam/embedding_6/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_6/embeddings/m*
_output_shapes
:	?d*
dtype0
?
Adam/embedding_7/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*.
shared_nameAdam/embedding_7/embeddings/m
?
1Adam/embedding_7/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/m*
_output_shapes
:	?d*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/m
?
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_6/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*.
shared_nameAdam/embedding_6/embeddings/v
?
1Adam/embedding_6/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_6/embeddings/v*
_output_shapes
:	?d*
dtype0
?
Adam/embedding_7/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*.
shared_nameAdam/embedding_7/embeddings/v
?
1Adam/embedding_7/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/v*
_output_shapes
:	?d*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_7/kernel/v
?
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?,
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemdme&mf'mg,mh-mivjvk&vl'vm,vn-vo
 
*
0
1
&2
'3
,4
-5
*
0
1
&2
'3
,4
-5
?
7non_trainable_variables
regularization_losses

8layers
	variables
9metrics
:layer_metrics
;layer_regularization_losses
trainable_variables
 
fd
VARIABLE_VALUEembedding_6/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
<non_trainable_variables
regularization_losses

=layers
	variables
>metrics
?layer_metrics
@layer_regularization_losses
trainable_variables
fd
VARIABLE_VALUEembedding_7/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
Anon_trainable_variables
regularization_losses

Blayers
	variables
Cmetrics
Dlayer_metrics
Elayer_regularization_losses
trainable_variables
 
 
 
?
Fnon_trainable_variables
regularization_losses

Glayers
	variables
Hmetrics
Ilayer_metrics
Jlayer_regularization_losses
trainable_variables
 
 
 
?
Knon_trainable_variables
regularization_losses

Llayers
	variables
Mmetrics
Nlayer_metrics
Olayer_regularization_losses
 trainable_variables
 
 
 
?
Pnon_trainable_variables
"regularization_losses

Qlayers
#	variables
Rmetrics
Slayer_metrics
Tlayer_regularization_losses
$trainable_variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
?
Unon_trainable_variables
(regularization_losses

Vlayers
)	variables
Wmetrics
Xlayer_metrics
Ylayer_regularization_losses
*trainable_variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
?
Znon_trainable_variables
.regularization_losses

[layers
/	variables
\metrics
]layer_metrics
^layer_regularization_losses
0trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

_0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	`total
	acount
b	variables
c	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

b	variables
??
VARIABLE_VALUEAdam/embedding_6/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_7/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_6/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_7/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_7Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_8Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_8embedding_7/embeddingsembedding_6/embeddingsdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_84568
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_6/embeddings/Read/ReadVariableOp*embedding_7/embeddings/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_6/embeddings/m/Read/ReadVariableOp1Adam/embedding_7/embeddings/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp1Adam/embedding_6/embeddings/v/Read/ReadVariableOp1Adam/embedding_7/embeddings/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_84890
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_6/embeddingsembedding_7/embeddingsdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding_6/embeddings/mAdam/embedding_7/embeddings/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/embedding_6/embeddings/vAdam/embedding_7/embeddings/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*%
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_84977??
?
t
H__inference_concatenate_3_layer_call_and_return_conditional_losses_84741
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_84778

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
r
H__inference_concatenate_3_layer_call_and_return_conditional_losses_84367

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
'__inference_model_3_layer_call_fn_84660
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_844832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?C
?

__inference__traced_save_84890
file_prefix5
1savev2_embedding_6_embeddings_read_readvariableop5
1savev2_embedding_7_embeddings_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_6_embeddings_m_read_readvariableop<
8savev2_adam_embedding_7_embeddings_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop<
8savev2_adam_embedding_6_embeddings_v_read_readvariableop<
8savev2_adam_embedding_7_embeddings_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fcfca7c077934b6a817d186340a9d02e/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_6_embeddings_read_readvariableop1savev2_embedding_7_embeddings_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_6_embeddings_m_read_readvariableop8savev2_adam_embedding_7_embeddings_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop8savev2_adam_embedding_6_embeddings_v_read_readvariableop8savev2_adam_embedding_7_embeddings_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?d:	?d:
??:?:	?:: : : : : : : :	?d:	?d:
??:?:	?::	?d:	?d:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?d:%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?d:%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?d:%!

_output_shapes
:	?d:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
|
'__inference_dense_6_layer_call_fn_84767

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_843872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_84718

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
'__inference_model_3_layer_call_fn_84678
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_845252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
#__inference_signature_wrapper_84568
input_7
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_842832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
'__inference_model_3_layer_call_fn_84498
input_7
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_844832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_84525

inputs
inputs_1
embedding_7_84505
embedding_6_84508
dense_6_84514
dense_6_84516
dense_7_84519
dense_7_84521
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_6/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_7_84505*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_842982%
#embedding_7/StatefulPartitionedCall?
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_6_84508*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_843202%
#embedding_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_843382
flatten_6/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall,embedding_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_843522
flatten_7/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_843672
concatenate_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_6_84514dense_6_84516*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_843872!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_84519dense_7_84521*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_844142!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_7_layer_call_and_return_conditional_losses_84298

inputs
embedding_lookup_84292
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_84292Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/84292*+
_output_shapes
:?????????d*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/84292*+
_output_shapes
:?????????d2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
F__inference_embedding_7_layer_call_and_return_conditional_losses_84705

inputs
embedding_lookup_84699
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_84699Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/84699*+
_output_shapes
:?????????d*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/84699*+
_output_shapes
:?????????d2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_84338

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_3_layer_call_fn_84747
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_843672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????d:?????????d:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
|
'__inference_dense_7_layer_call_fn_84787

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_844142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_flatten_7_layer_call_fn_84734

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_843522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_dense_7_layer_call_and_return_conditional_losses_84414

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?,
?
 __inference__wrapped_model_84283
input_7
input_8.
*model_3_embedding_7_embedding_lookup_84251.
*model_3_embedding_6_embedding_lookup_842572
.model_3_dense_6_matmul_readvariableop_resource3
/model_3_dense_6_biasadd_readvariableop_resource2
.model_3_dense_7_matmul_readvariableop_resource3
/model_3_dense_7_biasadd_readvariableop_resource
identity??
model_3/embedding_7/CastCastinput_8*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_3/embedding_7/Cast?
$model_3/embedding_7/embedding_lookupResourceGather*model_3_embedding_7_embedding_lookup_84251model_3/embedding_7/Cast:y:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_7/embedding_lookup/84251*+
_output_shapes
:?????????d*
dtype02&
$model_3/embedding_7/embedding_lookup?
-model_3/embedding_7/embedding_lookup/IdentityIdentity-model_3/embedding_7/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model_3/embedding_7/embedding_lookup/84251*+
_output_shapes
:?????????d2/
-model_3/embedding_7/embedding_lookup/Identity?
/model_3/embedding_7/embedding_lookup/Identity_1Identity6model_3/embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d21
/model_3/embedding_7/embedding_lookup/Identity_1?
model_3/embedding_6/CastCastinput_7*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_3/embedding_6/Cast?
$model_3/embedding_6/embedding_lookupResourceGather*model_3_embedding_6_embedding_lookup_84257model_3/embedding_6/Cast:y:0*
Tindices0*=
_class3
1/loc:@model_3/embedding_6/embedding_lookup/84257*+
_output_shapes
:?????????d*
dtype02&
$model_3/embedding_6/embedding_lookup?
-model_3/embedding_6/embedding_lookup/IdentityIdentity-model_3/embedding_6/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model_3/embedding_6/embedding_lookup/84257*+
_output_shapes
:?????????d2/
-model_3/embedding_6/embedding_lookup/Identity?
/model_3/embedding_6/embedding_lookup/Identity_1Identity6model_3/embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d21
/model_3/embedding_6/embedding_lookup/Identity_1?
model_3/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
model_3/flatten_6/Const?
model_3/flatten_6/ReshapeReshape8model_3/embedding_6/embedding_lookup/Identity_1:output:0 model_3/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_3/flatten_6/Reshape?
model_3/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
model_3/flatten_7/Const?
model_3/flatten_7/ReshapeReshape8model_3/embedding_7/embedding_lookup/Identity_1:output:0 model_3/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????d2
model_3/flatten_7/Reshape?
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axis?
model_3/concatenate_3/concatConcatV2"model_3/flatten_6/Reshape:output:0"model_3/flatten_7/Reshape:output:0*model_3/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model_3/concatenate_3/concat?
%model_3/dense_6/MatMul/ReadVariableOpReadVariableOp.model_3_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_3/dense_6/MatMul/ReadVariableOp?
model_3/dense_6/MatMulMatMul%model_3/concatenate_3/concat:output:0-model_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense_6/MatMul?
&model_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_3/dense_6/BiasAdd/ReadVariableOp?
model_3/dense_6/BiasAddBiasAdd model_3/dense_6/MatMul:product:0.model_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_3/dense_6/BiasAdd?
model_3/dense_6/ReluRelu model_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_3/dense_6/Relu?
%model_3/dense_7/MatMul/ReadVariableOpReadVariableOp.model_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_3/dense_7/MatMul/ReadVariableOp?
model_3/dense_7/MatMulMatMul"model_3/dense_6/Relu:activations:0-model_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_7/MatMul?
&model_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/dense_7/BiasAdd/ReadVariableOp?
model_3/dense_7/BiasAddBiasAdd model_3/dense_7/MatMul:product:0.model_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/dense_7/BiasAdd?
model_3/dense_7/ReluRelu model_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/dense_7/Reluv
IdentityIdentity"model_3/dense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?'
?
B__inference_model_3_layer_call_and_return_conditional_losses_84605
inputs_0
inputs_1&
"embedding_7_embedding_lookup_84573&
"embedding_6_embedding_lookup_84579*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity?w
embedding_7/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_7/Cast?
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_84573embedding_7/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/84573*+
_output_shapes
:?????????d*
dtype02
embedding_7/embedding_lookup?
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/84573*+
_output_shapes
:?????????d2'
%embedding_7/embedding_lookup/Identity?
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_7/embedding_lookup/Identity_1w
embedding_6/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_6/Cast?
embedding_6/embedding_lookupResourceGather"embedding_6_embedding_lookup_84579embedding_6/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_6/embedding_lookup/84579*+
_output_shapes
:?????????d*
dtype02
embedding_6/embedding_lookup?
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_6/embedding_lookup/84579*+
_output_shapes
:?????????d2'
%embedding_6/embedding_lookup/Identity?
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_6/embedding_lookup/Identity_1s
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten_6/Const?
flatten_6/ReshapeReshape0embedding_6/embedding_lookup/Identity_1:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten_6/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten_7/Const?
flatten_7/ReshapeReshape0embedding_7/embedding_lookup/Identity_1:output:0flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten_7/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2flatten_6/Reshape:output:0flatten_7/Reshape:output:0"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_3/concat?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulconcatenate_3/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Relun
IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_6_layer_call_and_return_conditional_losses_84387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?p
?
!__inference__traced_restore_84977
file_prefix+
'assignvariableop_embedding_6_embeddings-
)assignvariableop_1_embedding_7_embeddings%
!assignvariableop_2_dense_6_kernel#
assignvariableop_3_dense_6_bias%
!assignvariableop_4_dense_7_kernel#
assignvariableop_5_dense_7_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count5
1assignvariableop_13_adam_embedding_6_embeddings_m5
1assignvariableop_14_adam_embedding_7_embeddings_m-
)assignvariableop_15_adam_dense_6_kernel_m+
'assignvariableop_16_adam_dense_6_bias_m-
)assignvariableop_17_adam_dense_7_kernel_m+
'assignvariableop_18_adam_dense_7_bias_m5
1assignvariableop_19_adam_embedding_6_embeddings_v5
1assignvariableop_20_adam_embedding_7_embeddings_v-
)assignvariableop_21_adam_dense_6_kernel_v+
'assignvariableop_22_adam_dense_6_bias_v-
)assignvariableop_23_adam_dense_7_kernel_v+
'assignvariableop_24_adam_dense_7_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_6_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_7_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_6_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_6_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_7_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_7_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_embedding_6_embeddings_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_embedding_7_embeddings_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_6_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_6_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_7_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_7_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_6_embeddings_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_embedding_7_embeddings_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_6_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_6_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_7_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_7_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_flatten_6_layer_call_fn_84723

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_843382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_84455
input_7
input_8
embedding_7_84435
embedding_6_84438
dense_6_84444
dense_6_84446
dense_7_84449
dense_7_84451
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_6/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinput_8embedding_7_84435*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_842982%
#embedding_7/StatefulPartitionedCall?
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinput_7embedding_6_84438*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_843202%
#embedding_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_843382
flatten_6/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall,embedding_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_843522
flatten_7/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_843672
concatenate_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_6_84444dense_6_84446*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_843872!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_84449dense_7_84451*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_844142!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_84729

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_84483

inputs
inputs_1
embedding_7_84463
embedding_6_84466
dense_6_84472
dense_6_84474
dense_7_84477
dense_7_84479
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_6/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_7_84463*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_842982%
#embedding_7/StatefulPartitionedCall?
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_6_84466*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_843202%
#embedding_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_843382
flatten_6/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall,embedding_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_843522
flatten_7/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_843672
concatenate_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_6_84472dense_6_84474*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_843872!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_84477dense_7_84479*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_844142!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_6_layer_call_and_return_conditional_losses_84688

inputs
embedding_lookup_84682
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_84682Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/84682*+
_output_shapes
:?????????d*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/84682*+
_output_shapes
:?????????d2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
q
+__inference_embedding_6_layer_call_fn_84695

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_843202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
'__inference_model_3_layer_call_fn_84540
input_7
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_845252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
q
+__inference_embedding_7_layer_call_fn_84712

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_842982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
B__inference_dense_6_layer_call_and_return_conditional_losses_84758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_embedding_6_layer_call_and_return_conditional_losses_84320

inputs
embedding_lookup_84314
identity?]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_84314Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/84314*+
_output_shapes
:?????????d*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/84314*+
_output_shapes
:?????????d2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
`
D__inference_flatten_7_layer_call_and_return_conditional_losses_84352

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????d2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????d:S O
+
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_84431
input_7
input_8
embedding_7_84307
embedding_6_84329
dense_6_84398
dense_6_84400
dense_7_84425
dense_7_84427
identity??dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?#embedding_6/StatefulPartitionedCall?#embedding_7/StatefulPartitionedCall?
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinput_8embedding_7_84307*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_842982%
#embedding_7/StatefulPartitionedCall?
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinput_7embedding_6_84329*
Tin
2*
Tout
2*+
_output_shapes
:?????????d*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_843202%
#embedding_6/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_843382
flatten_6/PartitionedCall?
flatten_7/PartitionedCallPartitionedCall,embedding_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_7_layer_call_and_return_conditional_losses_843522
flatten_7/PartitionedCall?
concatenate_3/PartitionedCallPartitionedCall"flatten_6/PartitionedCall:output:0"flatten_7/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_843672
concatenate_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_6_84398dense_6_84400*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_843872!
dense_6/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_84425dense_7_84427*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_844142!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?'
?
B__inference_model_3_layer_call_and_return_conditional_losses_84642
inputs_0
inputs_1&
"embedding_7_embedding_lookup_84610&
"embedding_6_embedding_lookup_84616*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity?w
embedding_7/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_7/Cast?
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_84610embedding_7/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/84610*+
_output_shapes
:?????????d*
dtype02
embedding_7/embedding_lookup?
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/84610*+
_output_shapes
:?????????d2'
%embedding_7/embedding_lookup/Identity?
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_7/embedding_lookup/Identity_1w
embedding_6/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_6/Cast?
embedding_6/embedding_lookupResourceGather"embedding_6_embedding_lookup_84616embedding_6/Cast:y:0*
Tindices0*5
_class+
)'loc:@embedding_6/embedding_lookup/84616*+
_output_shapes
:?????????d*
dtype02
embedding_6/embedding_lookup?
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_6/embedding_lookup/84616*+
_output_shapes
:?????????d2'
%embedding_6/embedding_lookup/Identity?
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d2)
'embedding_6/embedding_lookup/Identity_1s
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten_6/Const?
flatten_6/ReshapeReshape0embedding_6/embedding_lookup/Identity_1:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten_6/Reshapes
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   2
flatten_7/Const?
flatten_7/ReshapeReshape0embedding_7/embedding_lookup/Identity_1:output:0flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????d2
flatten_7/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axis?
concatenate_3/concatConcatV2flatten_6/Reshape:output:0flatten_7/Reshape:output:0"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatenate_3/concat?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulconcatenate_3/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Relun
IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:?????????:?????????:::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_70
serving_default_input_7:0?????????
;
input_80
serving_default_input_8:0?????????;
dense_70
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?<
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*p&call_and_return_all_conditional_losses
q__call__
r_default_save_signature"?8
_tf_keras_model?8{"class_name": "Model", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1518, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1636, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["embedding_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["embedding_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_6", 0, 0, {}], ["flatten_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1518, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1636, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["embedding_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["embedding_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_6", 0, 0, {}], ["flatten_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1518, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 1636, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
regularization_losses
	variables
 trainable_variables
!	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100]}, {"class_name": "TensorShape", "items": [null, 100]}]}
?

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?

,kernel
-bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
*&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemdme&mf'mg,mh-mivjvk&vl'vm,vn-vo"
	optimizer
 "
trackable_list_wrapper
J
0
1
&2
'3
,4
-5"
trackable_list_wrapper
J
0
1
&2
'3
,4
-5"
trackable_list_wrapper
?
7non_trainable_variables
regularization_losses

8layers
	variables
9metrics
:layer_metrics
;layer_regularization_losses
trainable_variables
q__call__
r_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'	?d2embedding_6/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
<non_trainable_variables
regularization_losses

=layers
	variables
>metrics
?layer_metrics
@layer_regularization_losses
trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
):'	?d2embedding_7/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
Anon_trainable_variables
regularization_losses

Blayers
	variables
Cmetrics
Dlayer_metrics
Elayer_regularization_losses
trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables
regularization_losses

Glayers
	variables
Hmetrics
Ilayer_metrics
Jlayer_regularization_losses
trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables
regularization_losses

Llayers
	variables
Mmetrics
Nlayer_metrics
Olayer_regularization_losses
 trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pnon_trainable_variables
"regularization_losses

Qlayers
#	variables
Rmetrics
Slayer_metrics
Tlayer_regularization_losses
$trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
Unon_trainable_variables
(regularization_losses

Vlayers
)	variables
Wmetrics
Xlayer_metrics
Ylayer_regularization_losses
*trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
Znon_trainable_variables
.regularization_losses

[layers
/	variables
\metrics
]layer_metrics
^layer_regularization_losses
0trainable_variables
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	`total
	acount
b	variables
c	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
.:,	?d2Adam/embedding_6/embeddings/m
.:,	?d2Adam/embedding_7/embeddings/m
':%
??2Adam/dense_6/kernel/m
 :?2Adam/dense_6/bias/m
&:$	?2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
.:,	?d2Adam/embedding_6/embeddings/v
.:,	?d2Adam/embedding_7/embeddings/v
':%
??2Adam/dense_6/kernel/v
 :?2Adam/dense_6/bias/v
&:$	?2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
?2?
B__inference_model_3_layer_call_and_return_conditional_losses_84605
B__inference_model_3_layer_call_and_return_conditional_losses_84455
B__inference_model_3_layer_call_and_return_conditional_losses_84642
B__inference_model_3_layer_call_and_return_conditional_losses_84431?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_3_layer_call_fn_84498
'__inference_model_3_layer_call_fn_84540
'__inference_model_3_layer_call_fn_84678
'__inference_model_3_layer_call_fn_84660?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_84283?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_7?????????
!?
input_8?????????
?2?
F__inference_embedding_6_layer_call_and_return_conditional_losses_84688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embedding_6_layer_call_fn_84695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_7_layer_call_and_return_conditional_losses_84705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embedding_7_layer_call_fn_84712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_6_layer_call_and_return_conditional_losses_84718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_6_layer_call_fn_84723?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_7_layer_call_and_return_conditional_losses_84729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_7_layer_call_fn_84734?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_84741?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_3_layer_call_fn_84747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_6_layer_call_and_return_conditional_losses_84758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_6_layer_call_fn_84767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_7_layer_call_and_return_conditional_losses_84778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_7_layer_call_fn_84787?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
9B7
#__inference_signature_wrapper_84568input_7input_8?
 __inference__wrapped_model_84283?&',-X?U
N?K
I?F
!?
input_7?????????
!?
input_8?????????
? "1?.
,
dense_7!?
dense_7??????????
H__inference_concatenate_3_layer_call_and_return_conditional_losses_84741?Z?W
P?M
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d
? "&?#
?
0??????????
? ?
-__inference_concatenate_3_layer_call_fn_84747wZ?W
P?M
K?H
"?
inputs/0?????????d
"?
inputs/1?????????d
? "????????????
B__inference_dense_6_layer_call_and_return_conditional_losses_84758^&'0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_6_layer_call_fn_84767Q&'0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_7_layer_call_and_return_conditional_losses_84778],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_7_layer_call_fn_84787P,-0?-
&?#
!?
inputs??????????
? "???????????
F__inference_embedding_6_layer_call_and_return_conditional_losses_84688_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????d
? ?
+__inference_embedding_6_layer_call_fn_84695R/?,
%?"
 ?
inputs?????????
? "??????????d?
F__inference_embedding_7_layer_call_and_return_conditional_losses_84705_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????d
? ?
+__inference_embedding_7_layer_call_fn_84712R/?,
%?"
 ?
inputs?????????
? "??????????d?
D__inference_flatten_6_layer_call_and_return_conditional_losses_84718\3?0
)?&
$?!
inputs?????????d
? "%?"
?
0?????????d
? |
)__inference_flatten_6_layer_call_fn_84723O3?0
)?&
$?!
inputs?????????d
? "??????????d?
D__inference_flatten_7_layer_call_and_return_conditional_losses_84729\3?0
)?&
$?!
inputs?????????d
? "%?"
?
0?????????d
? |
)__inference_flatten_7_layer_call_fn_84734O3?0
)?&
$?!
inputs?????????d
? "??????????d?
B__inference_model_3_layer_call_and_return_conditional_losses_84431?&',-`?]
V?S
I?F
!?
input_7?????????
!?
input_8?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_84455?&',-`?]
V?S
I?F
!?
input_7?????????
!?
input_8?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_84605?&',-b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_84642?&',-b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
'__inference_model_3_layer_call_fn_84498?&',-`?]
V?S
I?F
!?
input_7?????????
!?
input_8?????????
p

 
? "???????????
'__inference_model_3_layer_call_fn_84540?&',-`?]
V?S
I?F
!?
input_7?????????
!?
input_8?????????
p 

 
? "???????????
'__inference_model_3_layer_call_fn_84660?&',-b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
'__inference_model_3_layer_call_fn_84678?&',-b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
#__inference_signature_wrapper_84568?&',-i?f
? 
_?\
,
input_7!?
input_7?????????
,
input_8!?
input_8?????????"1?.
,
dense_7!?
dense_7?????????