       �K"	   RI��Abrain.Event:2e�ɷ�r      ��	f� RI��A"��
Y
Inputs/xPlaceholder*
_output_shapes

:d*
shape
:d*
dtype0
Y
Inputs/yPlaceholder*
dtype0*
shape
:d*
_output_shapes

:d
�
BNet/hidden_layer/kernel/Initializer/stateless_random_uniform/shapeConst*
dtype0*
_output_shapes
:**
_class 
loc:@Net/hidden_layer/kernel*
valueB"   
   
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/minConst*
valueB
 *�=�*
_output_shapes
: **
_class 
loc:@Net/hidden_layer/kernel*
dtype0
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/maxConst*
valueB
 *�=?**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes
: *
dtype0
�
^Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
valueB" ��    *
dtype0*
_output_shapes
:**
_class 
loc:@Net/hidden_layer/kernel
�
YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter^Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed**
_class 
loc:@Net/hidden_layer/kernel* 
_output_shapes
::*
Tseed0
�
YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
value	B :*
_output_shapes
: *
dtype0**
_class 
loc:@Net/hidden_layer/kernel
�
UNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2BNet/hidden_layer/kernel/Initializer/stateless_random_uniform/shapeYNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter[Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes

:
*
dtype0*
Tshape0
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/subSub@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/max@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/min**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes
: *
T0
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/mulMulUNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/sub*
_output_shapes

:
**
_class 
loc:@Net/hidden_layer/kernel*
T0
�
<Net/hidden_layer/kernel/Initializer/stateless_random_uniformAddV2@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/mul@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/min*
T0*
_output_shapes

:
**
_class 
loc:@Net/hidden_layer/kernel
�
Net/hidden_layer/kernelVarHandleOp*
dtype0*(
shared_nameNet/hidden_layer/kernel**
_class 
loc:@Net/hidden_layer/kernel*
	container *
_output_shapes
: *
allowed_devices
 *
shape
:


8Net/hidden_layer/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/hidden_layer/kernel*
_output_shapes
: 
�
Net/hidden_layer/kernel/AssignAssignVariableOpNet/hidden_layer/kernel<Net/hidden_layer/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
�
+Net/hidden_layer/kernel/Read/ReadVariableOpReadVariableOpNet/hidden_layer/kernel*
_output_shapes

:
*
dtype0
�
'Net/hidden_layer/bias/Initializer/zerosConst*
dtype0*
valueB
*    *
_output_shapes
:
*(
_class
loc:@Net/hidden_layer/bias
�
Net/hidden_layer/biasVarHandleOp*
_output_shapes
: *
allowed_devices
 *
shape:
*(
_class
loc:@Net/hidden_layer/bias*
	container *
dtype0*&
shared_nameNet/hidden_layer/bias
{
6Net/hidden_layer/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/hidden_layer/bias*
_output_shapes
: 
�
Net/hidden_layer/bias/AssignAssignVariableOpNet/hidden_layer/bias'Net/hidden_layer/bias/Initializer/zeros*
dtype0*
validate_shape( 
{
)Net/hidden_layer/bias/Read/ReadVariableOpReadVariableOpNet/hidden_layer/bias*
dtype0*
_output_shapes
:

~
&Net/hidden_layer/MatMul/ReadVariableOpReadVariableOpNet/hidden_layer/kernel*
dtype0*
_output_shapes

:

�
Net/hidden_layer/MatMulMatMulInputs/x&Net/hidden_layer/MatMul/ReadVariableOp*
_output_shapes

:d
*
transpose_b( *
transpose_a( *
T0
y
'Net/hidden_layer/BiasAdd/ReadVariableOpReadVariableOpNet/hidden_layer/bias*
_output_shapes
:
*
dtype0
�
Net/hidden_layer/BiasAddBiasAddNet/hidden_layer/MatMul'Net/hidden_layer/BiasAdd/ReadVariableOp*
_output_shapes

:d
*
T0*
data_formatNHWC
`
Net/hidden_layer/ReluReluNet/hidden_layer/BiasAdd*
T0*
_output_shapes

:d

�
BNet/output_layer/kernel/Initializer/stateless_random_uniform/shapeConst*
valueB"
      *
_output_shapes
:**
_class 
loc:@Net/output_layer/kernel*
dtype0
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/minConst*
_output_shapes
: **
_class 
loc:@Net/output_layer/kernel*
valueB
 *�=�*
dtype0
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/maxConst*
valueB
 *�=?**
_class 
loc:@Net/output_layer/kernel*
_output_shapes
: *
dtype0
�
^Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0**
_class 
loc:@Net/output_layer/kernel*
valueB"�;�    
�
YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter^Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*
Tseed0**
_class 
loc:@Net/output_layer/kernel* 
_output_shapes
::
�
YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst**
_class 
loc:@Net/output_layer/kernel*
value	B :*
_output_shapes
: *
dtype0
�
UNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2BNet/output_layer/kernel/Initializer/stateless_random_uniform/shapeYNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter[Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
_output_shapes

:
*
Tshape0**
_class 
loc:@Net/output_layer/kernel
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/subSub@Net/output_layer/kernel/Initializer/stateless_random_uniform/max@Net/output_layer/kernel/Initializer/stateless_random_uniform/min*
_output_shapes
: *
T0**
_class 
loc:@Net/output_layer/kernel
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/mulMulUNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2@Net/output_layer/kernel/Initializer/stateless_random_uniform/sub*
T0*
_output_shapes

:
**
_class 
loc:@Net/output_layer/kernel
�
<Net/output_layer/kernel/Initializer/stateless_random_uniformAddV2@Net/output_layer/kernel/Initializer/stateless_random_uniform/mul@Net/output_layer/kernel/Initializer/stateless_random_uniform/min*
T0*
_output_shapes

:
**
_class 
loc:@Net/output_layer/kernel
�
Net/output_layer/kernelVarHandleOp**
_class 
loc:@Net/output_layer/kernel*
dtype0*(
shared_nameNet/output_layer/kernel*
_output_shapes
: *
allowed_devices
 *
shape
:
*
	container 

8Net/output_layer/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/output_layer/kernel*
_output_shapes
: 
�
Net/output_layer/kernel/AssignAssignVariableOpNet/output_layer/kernel<Net/output_layer/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
�
+Net/output_layer/kernel/Read/ReadVariableOpReadVariableOpNet/output_layer/kernel*
dtype0*
_output_shapes

:

�
'Net/output_layer/bias/Initializer/zerosConst*(
_class
loc:@Net/output_layer/bias*
valueB*    *
_output_shapes
:*
dtype0
�
Net/output_layer/biasVarHandleOp*
shape:*
allowed_devices
 *
dtype0*(
_class
loc:@Net/output_layer/bias*
_output_shapes
: *&
shared_nameNet/output_layer/bias*
	container 
{
6Net/output_layer/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/output_layer/bias*
_output_shapes
: 
�
Net/output_layer/bias/AssignAssignVariableOpNet/output_layer/bias'Net/output_layer/bias/Initializer/zeros*
dtype0*
validate_shape( 
{
)Net/output_layer/bias/Read/ReadVariableOpReadVariableOpNet/output_layer/bias*
_output_shapes
:*
dtype0
~
&Net/output_layer/MatMul/ReadVariableOpReadVariableOpNet/output_layer/kernel*
dtype0*
_output_shapes

:

�
Net/output_layer/MatMulMatMulNet/hidden_layer/Relu&Net/output_layer/MatMul/ReadVariableOp*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:d
y
'Net/output_layer/BiasAdd/ReadVariableOpReadVariableOpNet/output_layer/bias*
_output_shapes
:*
dtype0
�
Net/output_layer/BiasAddBiasAddNet/output_layer/MatMul'Net/output_layer/BiasAdd/ReadVariableOp*
data_formatNHWC*
T0*
_output_shapes

:d
W
Net/h_out/tagConst*
_output_shapes
: *
dtype0*
valueB B	Net/h_out
d
	Net/h_outHistogramSummaryNet/h_out/tagNet/hidden_layer/Relu*
_output_shapes
: *
T0
U
Net/pred/tagConst*
valueB BNet/pred*
dtype0*
_output_shapes
: 
e
Net/predHistogramSummaryNet/pred/tagNet/output_layer/BiasAdd*
T0*
_output_shapes
: 
x
loss/SquaredDifferenceSquaredDifferenceNet/output_layer/BiasAddInputs/y*
_output_shapes

:d*
T0
f
!loss/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
j
'loss/assert_broadcastable/weights/shapeConst*
valueB *
_output_shapes
: *
dtype0
h
&loss/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B : *
dtype0
w
&loss/assert_broadcastable/values/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      
g
%loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
=
5loss/assert_broadcastable/static_scalar_check_successNoOp
�
loss/Cast/xConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *  �?*
dtype0
]
loss/MulMulloss/SquaredDifferenceloss/Cast/x*
_output_shapes

:d*
T0
�

loss/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
valueB"       *
dtype0
c
loss/SumSumloss/Mul
loss/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
�
loss/num_present/Equal/yConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
loss/num_present/EqualEqualloss/Cast/xloss/num_present/Equal/y*
T0*
_output_shapes
: *
incompatible_shape_error(
�
loss/num_present/zeros_likeConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB
 *    *
dtype0
�
0loss/num_present/ones_like/Shape/shape_as_tensorConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB 
�
 loss/num_present/ones_like/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
loss/num_present/ones_likeFill0loss/num_present/ones_like/Shape/shape_as_tensor loss/num_present/ones_like/Const*
_output_shapes
: *
T0*

index_type0
�
loss/num_present/SelectSelectloss/num_present/Equalloss/num_present/zeros_likeloss/num_present/ones_like*
_output_shapes
: *
T0
�
Eloss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
�
Dloss/num_present/broadcast_weights/assert_broadcastable/weights/rankConst6^loss/assert_broadcastable/static_scalar_check_success*
value	B : *
_output_shapes
: *
dtype0
�
Dloss/num_present/broadcast_weights/assert_broadcastable/values/shapeConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"d      *
_output_shapes
:
�
Closs/num_present/broadcast_weights/assert_broadcastable/values/rankConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
value	B :*
dtype0
�
Sloss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp6^loss/assert_broadcastable/static_scalar_check_success
�
Bloss/num_present/broadcast_weights/ones_like/Shape/shape_as_tensorConst6^loss/assert_broadcastable/static_scalar_check_successT^loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"d      *
_output_shapes
:
�
2loss/num_present/broadcast_weights/ones_like/ConstConst6^loss/assert_broadcastable/static_scalar_check_successT^loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
,loss/num_present/broadcast_weights/ones_likeFillBloss/num_present/broadcast_weights/ones_like/Shape/shape_as_tensor2loss/num_present/broadcast_weights/ones_like/Const*

index_type0*
_output_shapes

:d*
T0
�
"loss/num_present/broadcast_weightsMulloss/num_present/Select,loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes

:d
�
loss/num_present/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
_output_shapes
:*
dtype0
�
loss/num_presentSum"loss/num_present/broadcast_weightsloss/num_present/Const*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
�
	loss/RankConst6^loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
loss/range/startConst6^loss/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
�
loss/range/deltaConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B :
h

loss/rangeRangeloss/range/start	loss/Rankloss/range/delta*

Tidx0*
_output_shapes
: 
e

loss/Sum_1Sumloss/Sum
loss/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
U

loss/valueDivNoNan
loss/Sum_1loss/num_present*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
^
gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
_output_shapes
: *
T0*

index_type0
b
gradients/loss/value_grad/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
d
!gradients/loss/value_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
�
/gradients/loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/value_grad/Shape!gradients/loss/value_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
x
$gradients/loss/value_grad/div_no_nanDivNoNangradients/grad_ys_0loss/num_present*
_output_shapes
: *
T0
�
gradients/loss/value_grad/SumSum$gradients/loss/value_grad/div_no_nan/gradients/loss/value_grad/BroadcastGradientArgs*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
!gradients/loss/value_grad/ReshapeReshapegradients/loss/value_grad/Sumgradients/loss/value_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Q
gradients/loss/value_grad/NegNeg
loss/Sum_1*
T0*
_output_shapes
: 
�
&gradients/loss/value_grad/div_no_nan_1DivNoNangradients/loss/value_grad/Negloss/num_present*
T0*
_output_shapes
: 
�
&gradients/loss/value_grad/div_no_nan_2DivNoNan&gradients/loss/value_grad/div_no_nan_1loss/num_present*
_output_shapes
: *
T0
�
gradients/loss/value_grad/mulMulgradients/grad_ys_0&gradients/loss/value_grad/div_no_nan_2*
_output_shapes
: *
T0
�
gradients/loss/value_grad/Sum_1Sumgradients/loss/value_grad/mul1gradients/loss/value_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
�
#gradients/loss/value_grad/Reshape_1Reshapegradients/loss/value_grad/Sum_1!gradients/loss/value_grad/Shape_1*
_output_shapes
: *
Tshape0*
T0
|
*gradients/loss/value_grad/tuple/group_depsNoOp"^gradients/loss/value_grad/Reshape$^gradients/loss/value_grad/Reshape_1
�
2gradients/loss/value_grad/tuple/control_dependencyIdentity!gradients/loss/value_grad/Reshape+^gradients/loss/value_grad/tuple/group_deps*
_output_shapes
: *4
_class*
(&loc:@gradients/loss/value_grad/Reshape*
T0
�
4gradients/loss/value_grad/tuple/control_dependency_1Identity#gradients/loss/value_grad/Reshape_1+^gradients/loss/value_grad/tuple/group_deps*6
_class,
*(loc:@gradients/loss/value_grad/Reshape_1*
T0*
_output_shapes
: 
j
'gradients/loss/Sum_1_grad/Reshape/shapeConst*
_output_shapes
: *
valueB *
dtype0
l
)gradients/loss/Sum_1_grad/Reshape/shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
!gradients/loss/Sum_1_grad/ReshapeReshape2gradients/loss/value_grad/tuple/control_dependency)gradients/loss/Sum_1_grad/Reshape/shape_1*
_output_shapes
: *
Tshape0*
T0
b
gradients/loss/Sum_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 
�
gradients/loss/Sum_1_grad/TileTile!gradients/loss/Sum_1_grad/Reshapegradients/loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
v
%gradients/loss/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
gradients/loss/Sum_grad/ReshapeReshapegradients/loss/Sum_1_grad/Tile%gradients/loss/Sum_grad/Reshape/shape*
_output_shapes

:*
Tshape0*
T0
n
gradients/loss/Sum_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB"d      
�
gradients/loss/Sum_grad/TileTilegradients/loss/Sum_grad/Reshapegradients/loss/Sum_grad/Const*
_output_shapes

:d*

Tmultiples0*
T0
�
0gradients/loss/Mul_grad/BroadcastGradientArgs/s0Const*
valueB"d      *
_output_shapes
:*
dtype0
s
0gradients/loss/Mul_grad/BroadcastGradientArgs/s1Const*
valueB *
dtype0*
_output_shapes
: 
�
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/loss/Mul_grad/BroadcastGradientArgs/s00gradients/loss/Mul_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:���������:���������*
T0
v
gradients/loss/Mul_grad/MulMulgradients/loss/Sum_grad/Tileloss/Cast/x*
T0*
_output_shapes

:d
�
gradients/loss/Mul_grad/Mul_1Mulloss/SquaredDifferencegradients/loss/Sum_grad/Tile*
_output_shapes

:d*
T0
~
-gradients/loss/Mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
valueB"       *
dtype0
�
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/Mul_1-gradients/loss/Mul_grad/Sum/reduction_indices*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
h
%gradients/loss/Mul_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
j
'gradients/loss/Mul_grad/Reshape/shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sum'gradients/loss/Mul_grad/Reshape/shape_1*
Tshape0*
T0*
_output_shapes
: 
p
(gradients/loss/Mul_grad/tuple/group_depsNoOp^gradients/loss/Mul_grad/Mul ^gradients/loss/Mul_grad/Reshape
�
0gradients/loss/Mul_grad/tuple/control_dependencyIdentitygradients/loss/Mul_grad/Mul)^gradients/loss/Mul_grad/tuple/group_deps*.
_class$
" loc:@gradients/loss/Mul_grad/Mul*
T0*
_output_shapes

:d
�
2gradients/loss/Mul_grad/tuple/control_dependency_1Identitygradients/loss/Mul_grad/Reshape)^gradients/loss/Mul_grad/tuple/group_deps*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients/loss/Mul_grad/Reshape
�
,gradients/loss/SquaredDifference_grad/scalarConst1^gradients/loss/Mul_grad/tuple/control_dependency*
_output_shapes
: *
valueB
 *   @*
dtype0
�
)gradients/loss/SquaredDifference_grad/MulMul,gradients/loss/SquaredDifference_grad/scalar0gradients/loss/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:d
�
)gradients/loss/SquaredDifference_grad/subSubNet/output_layer/BiasAddInputs/y1^gradients/loss/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:d
�
+gradients/loss/SquaredDifference_grad/mul_1Mul)gradients/loss/SquaredDifference_grad/Mul)gradients/loss/SquaredDifference_grad/sub*
_output_shapes

:d*
T0
�
)gradients/loss/SquaredDifference_grad/NegNeg+gradients/loss/SquaredDifference_grad/mul_1*
_output_shapes

:d*
T0
�
6gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp*^gradients/loss/SquaredDifference_grad/Neg,^gradients/loss/SquaredDifference_grad/mul_1
�
>gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity+gradients/loss/SquaredDifference_grad/mul_17^gradients/loss/SquaredDifference_grad/tuple/group_deps*>
_class4
20loc:@gradients/loss/SquaredDifference_grad/mul_1*
_output_shapes

:d*
T0
�
@gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity)gradients/loss/SquaredDifference_grad/Neg7^gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*
_output_shapes

:d*<
_class2
0.loc:@gradients/loss/SquaredDifference_grad/Neg
�
3gradients/Net/output_layer/BiasAdd_grad/BiasAddGradBiasAddGrad>gradients/loss/SquaredDifference_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
:*
T0
�
8gradients/Net/output_layer/BiasAdd_grad/tuple/group_depsNoOp4^gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad?^gradients/loss/SquaredDifference_grad/tuple/control_dependency
�
@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependencyIdentity>gradients/loss/SquaredDifference_grad/tuple/control_dependency9^gradients/Net/output_layer/BiasAdd_grad/tuple/group_deps*
_output_shapes

:d*
T0*>
_class4
20loc:@gradients/loss/SquaredDifference_grad/mul_1
�
Bgradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad9^gradients/Net/output_layer/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*F
_class<
:8loc:@gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad*
T0
�
-gradients/Net/output_layer/MatMul_grad/MatMulMatMul@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency&Net/output_layer/MatMul/ReadVariableOp*
T0*
_output_shapes

:d
*
transpose_b(*
transpose_a( 
�
/gradients/Net/output_layer/MatMul_grad/MatMul_1MatMulNet/hidden_layer/Relu@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:
*
transpose_b( *
transpose_a(
�
7gradients/Net/output_layer/MatMul_grad/tuple/group_depsNoOp.^gradients/Net/output_layer/MatMul_grad/MatMul0^gradients/Net/output_layer/MatMul_grad/MatMul_1
�
?gradients/Net/output_layer/MatMul_grad/tuple/control_dependencyIdentity-gradients/Net/output_layer/MatMul_grad/MatMul8^gradients/Net/output_layer/MatMul_grad/tuple/group_deps*
_output_shapes

:d
*@
_class6
42loc:@gradients/Net/output_layer/MatMul_grad/MatMul*
T0
�
Agradients/Net/output_layer/MatMul_grad/tuple/control_dependency_1Identity/gradients/Net/output_layer/MatMul_grad/MatMul_18^gradients/Net/output_layer/MatMul_grad/tuple/group_deps*
_output_shapes

:
*
T0*B
_class8
64loc:@gradients/Net/output_layer/MatMul_grad/MatMul_1
�
-gradients/Net/hidden_layer/Relu_grad/ReluGradReluGrad?gradients/Net/output_layer/MatMul_grad/tuple/control_dependencyNet/hidden_layer/Relu*
_output_shapes

:d
*
T0
�
3gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/Net/hidden_layer/Relu_grad/ReluGrad*
_output_shapes
:
*
T0*
data_formatNHWC
�
8gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_depsNoOp4^gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad.^gradients/Net/hidden_layer/Relu_grad/ReluGrad
�
@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/Net/hidden_layer/Relu_grad/ReluGrad9^gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes

:d
*@
_class6
42loc:@gradients/Net/hidden_layer/Relu_grad/ReluGrad
�
Bgradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad9^gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*F
_class<
:8loc:@gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad*
T0
�
-gradients/Net/hidden_layer/MatMul_grad/MatMulMatMul@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency&Net/hidden_layer/MatMul/ReadVariableOp*
transpose_b(*
T0*
_output_shapes

:d*
transpose_a( 
�
/gradients/Net/hidden_layer/MatMul_grad/MatMul_1MatMulInputs/x@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

:
*
T0
�
7gradients/Net/hidden_layer/MatMul_grad/tuple/group_depsNoOp.^gradients/Net/hidden_layer/MatMul_grad/MatMul0^gradients/Net/hidden_layer/MatMul_grad/MatMul_1
�
?gradients/Net/hidden_layer/MatMul_grad/tuple/control_dependencyIdentity-gradients/Net/hidden_layer/MatMul_grad/MatMul8^gradients/Net/hidden_layer/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Net/hidden_layer/MatMul_grad/MatMul*
_output_shapes

:d
�
Agradients/Net/hidden_layer/MatMul_grad/tuple/control_dependency_1Identity/gradients/Net/hidden_layer/MatMul_grad/MatMul_18^gradients/Net/hidden_layer/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Net/hidden_layer/MatMul_grad/MatMul_1*
_output_shapes

:

b
GradientDescent/learning_rateConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
�
KGradientDescent/update_Net/hidden_layer/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentNet/hidden_layer/kernelGradientDescent/learning_rateAgradients/Net/hidden_layer/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Net/hidden_layer/kernel
�
IGradientDescent/update_Net/hidden_layer/bias/ResourceApplyGradientDescentResourceApplyGradientDescentNet/hidden_layer/biasGradientDescent/learning_rateBgradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@Net/hidden_layer/bias
�
KGradientDescent/update_Net/output_layer/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentNet/output_layer/kernelGradientDescent/learning_rateAgradients/Net/output_layer/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Net/output_layer/kernel
�
IGradientDescent/update_Net/output_layer/bias/ResourceApplyGradientDescentResourceApplyGradientDescentNet/output_layer/biasGradientDescent/learning_rateBgradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@Net/output_layer/bias
�
GradientDescentNoOpJ^GradientDescent/update_Net/hidden_layer/bias/ResourceApplyGradientDescentL^GradientDescent/update_Net/hidden_layer/kernel/ResourceApplyGradientDescentJ^GradientDescent/update_Net/output_layer/bias/ResourceApplyGradientDescentL^GradientDescent/update_Net/output_layer/kernel/ResourceApplyGradientDescent
R
loss_1/tagsConst*
_output_shapes
: *
valueB Bloss_1*
dtype0
Q
loss_1ScalarSummaryloss_1/tags
loss/value*
T0*
_output_shapes
: 
�
initNoOp^Net/hidden_layer/bias/Assign^Net/hidden_layer/kernel/Assign^Net/output_layer/bias/Assign^Net/output_layer/kernel/Assign"�	�y�W]�      7��	LWRI��AJІ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
}
ResourceApplyGradientDescent
var

alpha"T

delta"T" 
Ttype:
2	"
use_lockingbool( �
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
�
StatelessRandomUniformV2
shape"Tshape
key
counter
alg
output"dtype"
dtypetype0:
2"
Tshapetype0:
2	
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�*2.10.02v2.10.0-rc3-6-g359c3cdfc5f��
Y
Inputs/xPlaceholder*
dtype0*
_output_shapes

:d*
shape
:d
Y
Inputs/yPlaceholder*
_output_shapes

:d*
shape
:d*
dtype0
�
BNet/hidden_layer/kernel/Initializer/stateless_random_uniform/shapeConst*
_output_shapes
:**
_class 
loc:@Net/hidden_layer/kernel*
dtype0*
valueB"   
   
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/minConst*
valueB
 *�=�*
_output_shapes
: **
_class 
loc:@Net/hidden_layer/kernel*
dtype0
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*
valueB
 *�=?**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes
: 
�
^Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_output_shapes
:*
dtype0*
valueB" ��    **
_class 
loc:@Net/hidden_layer/kernel
�
YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter^Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed* 
_output_shapes
::*
Tseed0**
_class 
loc:@Net/hidden_layer/kernel
�
YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*
value	B :**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes
: 
�
UNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2BNet/hidden_layer/kernel/Initializer/stateless_random_uniform/shapeYNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter[Net/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1YNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes

:

�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/subSub@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/max@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/min*
_output_shapes
: **
_class 
loc:@Net/hidden_layer/kernel*
T0
�
@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/mulMulUNet/hidden_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/sub**
_class 
loc:@Net/hidden_layer/kernel*
T0*
_output_shapes

:

�
<Net/hidden_layer/kernel/Initializer/stateless_random_uniformAddV2@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/mul@Net/hidden_layer/kernel/Initializer/stateless_random_uniform/min*
_output_shapes

:
*
T0**
_class 
loc:@Net/hidden_layer/kernel
�
Net/hidden_layer/kernelVarHandleOp*
allowed_devices
 *(
shared_nameNet/hidden_layer/kernel**
_class 
loc:@Net/hidden_layer/kernel*
_output_shapes
: *
	container *
shape
:
*
dtype0

8Net/hidden_layer/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/hidden_layer/kernel*
_output_shapes
: 
�
Net/hidden_layer/kernel/AssignAssignVariableOpNet/hidden_layer/kernel<Net/hidden_layer/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
�
+Net/hidden_layer/kernel/Read/ReadVariableOpReadVariableOpNet/hidden_layer/kernel*
_output_shapes

:
*
dtype0
�
'Net/hidden_layer/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
valueB
*    *(
_class
loc:@Net/hidden_layer/bias
�
Net/hidden_layer/biasVarHandleOp*
allowed_devices
 *
_output_shapes
: *
shape:
*
dtype0*&
shared_nameNet/hidden_layer/bias*(
_class
loc:@Net/hidden_layer/bias*
	container 
{
6Net/hidden_layer/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/hidden_layer/bias*
_output_shapes
: 
�
Net/hidden_layer/bias/AssignAssignVariableOpNet/hidden_layer/bias'Net/hidden_layer/bias/Initializer/zeros*
dtype0*
validate_shape( 
{
)Net/hidden_layer/bias/Read/ReadVariableOpReadVariableOpNet/hidden_layer/bias*
_output_shapes
:
*
dtype0
~
&Net/hidden_layer/MatMul/ReadVariableOpReadVariableOpNet/hidden_layer/kernel*
dtype0*
_output_shapes

:

�
Net/hidden_layer/MatMulMatMulInputs/x&Net/hidden_layer/MatMul/ReadVariableOp*
T0*
transpose_a( *
transpose_b( *
_output_shapes

:d

y
'Net/hidden_layer/BiasAdd/ReadVariableOpReadVariableOpNet/hidden_layer/bias*
_output_shapes
:
*
dtype0
�
Net/hidden_layer/BiasAddBiasAddNet/hidden_layer/MatMul'Net/hidden_layer/BiasAdd/ReadVariableOp*
_output_shapes

:d
*
T0*
data_formatNHWC
`
Net/hidden_layer/ReluReluNet/hidden_layer/BiasAdd*
T0*
_output_shapes

:d

�
BNet/output_layer/kernel/Initializer/stateless_random_uniform/shapeConst**
_class 
loc:@Net/output_layer/kernel*
_output_shapes
:*
dtype0*
valueB"
      
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/minConst*
dtype0*
valueB
 *�=�**
_class 
loc:@Net/output_layer/kernel*
_output_shapes
: 
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/maxConst**
_class 
loc:@Net/output_layer/kernel*
valueB
 *�=?*
_output_shapes
: *
dtype0
�
^Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0**
_class 
loc:@Net/output_layer/kernel*
valueB"�;�    *
_output_shapes
:
�
YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter^Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed* 
_output_shapes
::**
_class 
loc:@Net/output_layer/kernel*
Tseed0
�
YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0**
_class 
loc:@Net/output_layer/kernel*
value	B :
�
UNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2BNet/output_layer/kernel/Initializer/stateless_random_uniform/shapeYNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter[Net/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1YNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
_output_shapes

:
*
Tshape0**
_class 
loc:@Net/output_layer/kernel
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/subSub@Net/output_layer/kernel/Initializer/stateless_random_uniform/max@Net/output_layer/kernel/Initializer/stateless_random_uniform/min*
_output_shapes
: **
_class 
loc:@Net/output_layer/kernel*
T0
�
@Net/output_layer/kernel/Initializer/stateless_random_uniform/mulMulUNet/output_layer/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2@Net/output_layer/kernel/Initializer/stateless_random_uniform/sub**
_class 
loc:@Net/output_layer/kernel*
_output_shapes

:
*
T0
�
<Net/output_layer/kernel/Initializer/stateless_random_uniformAddV2@Net/output_layer/kernel/Initializer/stateless_random_uniform/mul@Net/output_layer/kernel/Initializer/stateless_random_uniform/min*
T0**
_class 
loc:@Net/output_layer/kernel*
_output_shapes

:

�
Net/output_layer/kernelVarHandleOp*
_output_shapes
: *
	container *(
shared_nameNet/output_layer/kernel**
_class 
loc:@Net/output_layer/kernel*
shape
:
*
dtype0*
allowed_devices
 

8Net/output_layer/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/output_layer/kernel*
_output_shapes
: 
�
Net/output_layer/kernel/AssignAssignVariableOpNet/output_layer/kernel<Net/output_layer/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
�
+Net/output_layer/kernel/Read/ReadVariableOpReadVariableOpNet/output_layer/kernel*
_output_shapes

:
*
dtype0
�
'Net/output_layer/bias/Initializer/zerosConst*
valueB*    *(
_class
loc:@Net/output_layer/bias*
dtype0*
_output_shapes
:
�
Net/output_layer/biasVarHandleOp*
allowed_devices
 *
	container *(
_class
loc:@Net/output_layer/bias*
shape:*
_output_shapes
: *
dtype0*&
shared_nameNet/output_layer/bias
{
6Net/output_layer/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpNet/output_layer/bias*
_output_shapes
: 
�
Net/output_layer/bias/AssignAssignVariableOpNet/output_layer/bias'Net/output_layer/bias/Initializer/zeros*
dtype0*
validate_shape( 
{
)Net/output_layer/bias/Read/ReadVariableOpReadVariableOpNet/output_layer/bias*
_output_shapes
:*
dtype0
~
&Net/output_layer/MatMul/ReadVariableOpReadVariableOpNet/output_layer/kernel*
_output_shapes

:
*
dtype0
�
Net/output_layer/MatMulMatMulNet/hidden_layer/Relu&Net/output_layer/MatMul/ReadVariableOp*
_output_shapes

:d*
transpose_b( *
transpose_a( *
T0
y
'Net/output_layer/BiasAdd/ReadVariableOpReadVariableOpNet/output_layer/bias*
dtype0*
_output_shapes
:
�
Net/output_layer/BiasAddBiasAddNet/output_layer/MatMul'Net/output_layer/BiasAdd/ReadVariableOp*
data_formatNHWC*
T0*
_output_shapes

:d
W
Net/h_out/tagConst*
valueB B	Net/h_out*
dtype0*
_output_shapes
: 
d
	Net/h_outHistogramSummaryNet/h_out/tagNet/hidden_layer/Relu*
_output_shapes
: *
T0
U
Net/pred/tagConst*
_output_shapes
: *
valueB BNet/pred*
dtype0
e
Net/predHistogramSummaryNet/pred/tagNet/output_layer/BiasAdd*
_output_shapes
: *
T0
x
loss/SquaredDifferenceSquaredDifferenceNet/output_layer/BiasAddInputs/y*
_output_shapes

:d*
T0
f
!loss/assert_broadcastable/weightsConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
j
'loss/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
h
&loss/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B : *
dtype0
w
&loss/assert_broadcastable/values/shapeConst*
_output_shapes
:*
valueB"d      *
dtype0
g
%loss/assert_broadcastable/values/rankConst*
_output_shapes
: *
value	B :*
dtype0
=
5loss/assert_broadcastable/static_scalar_check_successNoOp
�
loss/Cast/xConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  �?*
_output_shapes
: 
]
loss/MulMulloss/SquaredDifferenceloss/Cast/x*
_output_shapes

:d*
T0
�

loss/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
:*
valueB"       
c
loss/SumSumloss/Mul
loss/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
loss/num_present/Equal/yConst6^loss/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/num_present/EqualEqualloss/Cast/xloss/num_present/Equal/y*
_output_shapes
: *
incompatible_shape_error(*
T0
�
loss/num_present/zeros_likeConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
�
0loss/num_present/ones_like/Shape/shape_as_tensorConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
valueB *
dtype0
�
 loss/num_present/ones_like/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
loss/num_present/ones_likeFill0loss/num_present/ones_like/Shape/shape_as_tensor loss/num_present/ones_like/Const*
_output_shapes
: *
T0*

index_type0
�
loss/num_present/SelectSelectloss/num_present/Equalloss/num_present/zeros_likeloss/num_present/ones_like*
T0*
_output_shapes
: 
�
Eloss/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
�
Dloss/num_present/broadcast_weights/assert_broadcastable/weights/rankConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
Dloss/num_present/broadcast_weights/assert_broadcastable/values/shapeConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
dtype0*
valueB"d      
�
Closs/num_present/broadcast_weights/assert_broadcastable/values/rankConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B :
�
Sloss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp6^loss/assert_broadcastable/static_scalar_check_success
�
Bloss/num_present/broadcast_weights/ones_like/Shape/shape_as_tensorConst6^loss/assert_broadcastable/static_scalar_check_successT^loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"d      *
_output_shapes
:
�
2loss/num_present/broadcast_weights/ones_like/ConstConst6^loss/assert_broadcastable/static_scalar_check_successT^loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
valueB
 *  �?
�
,loss/num_present/broadcast_weights/ones_likeFillBloss/num_present/broadcast_weights/ones_like/Shape/shape_as_tensor2loss/num_present/broadcast_weights/ones_like/Const*

index_type0*
T0*
_output_shapes

:d
�
"loss/num_present/broadcast_weightsMulloss/num_present/Select,loss/num_present/broadcast_weights/ones_like*
T0*
_output_shapes

:d
�
loss/num_present/ConstConst6^loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
_output_shapes
:*
dtype0
�
loss/num_presentSum"loss/num_present/broadcast_weightsloss/num_present/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
	loss/RankConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
�
loss/range/startConst6^loss/assert_broadcastable/static_scalar_check_success*
_output_shapes
: *
dtype0*
value	B : 
�
loss/range/deltaConst6^loss/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B :
h

loss/rangeRangeloss/range/start	loss/Rankloss/range/delta*
_output_shapes
: *

Tidx0
e

loss/Sum_1Sumloss/Sum
loss/range*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
U

loss/valueDivNoNan
loss/Sum_1loss/num_present*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
^
gradients/grad_ys_0/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
z
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
_output_shapes
: *
T0*

index_type0
b
gradients/loss/value_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
d
!gradients/loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
/gradients/loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/loss/value_grad/Shape!gradients/loss/value_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
x
$gradients/loss/value_grad/div_no_nanDivNoNangradients/grad_ys_0loss/num_present*
_output_shapes
: *
T0
�
gradients/loss/value_grad/SumSum$gradients/loss/value_grad/div_no_nan/gradients/loss/value_grad/BroadcastGradientArgs*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
�
!gradients/loss/value_grad/ReshapeReshapegradients/loss/value_grad/Sumgradients/loss/value_grad/Shape*
T0*
_output_shapes
: *
Tshape0
Q
gradients/loss/value_grad/NegNeg
loss/Sum_1*
_output_shapes
: *
T0
�
&gradients/loss/value_grad/div_no_nan_1DivNoNangradients/loss/value_grad/Negloss/num_present*
T0*
_output_shapes
: 
�
&gradients/loss/value_grad/div_no_nan_2DivNoNan&gradients/loss/value_grad/div_no_nan_1loss/num_present*
T0*
_output_shapes
: 
�
gradients/loss/value_grad/mulMulgradients/grad_ys_0&gradients/loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
�
gradients/loss/value_grad/Sum_1Sumgradients/loss/value_grad/mul1gradients/loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
#gradients/loss/value_grad/Reshape_1Reshapegradients/loss/value_grad/Sum_1!gradients/loss/value_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
|
*gradients/loss/value_grad/tuple/group_depsNoOp"^gradients/loss/value_grad/Reshape$^gradients/loss/value_grad/Reshape_1
�
2gradients/loss/value_grad/tuple/control_dependencyIdentity!gradients/loss/value_grad/Reshape+^gradients/loss/value_grad/tuple/group_deps*
T0*
_output_shapes
: *4
_class*
(&loc:@gradients/loss/value_grad/Reshape
�
4gradients/loss/value_grad/tuple/control_dependency_1Identity#gradients/loss/value_grad/Reshape_1+^gradients/loss/value_grad/tuple/group_deps*6
_class,
*(loc:@gradients/loss/value_grad/Reshape_1*
_output_shapes
: *
T0
j
'gradients/loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
l
)gradients/loss/Sum_1_grad/Reshape/shape_1Const*
dtype0*
valueB *
_output_shapes
: 
�
!gradients/loss/Sum_1_grad/ReshapeReshape2gradients/loss/value_grad/tuple/control_dependency)gradients/loss/Sum_1_grad/Reshape/shape_1*
Tshape0*
_output_shapes
: *
T0
b
gradients/loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
�
gradients/loss/Sum_1_grad/TileTile!gradients/loss/Sum_1_grad/Reshapegradients/loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
v
%gradients/loss/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/loss/Sum_grad/ReshapeReshapegradients/loss/Sum_1_grad/Tile%gradients/loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
n
gradients/loss/Sum_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB"d      
�
gradients/loss/Sum_grad/TileTilegradients/loss/Sum_grad/Reshapegradients/loss/Sum_grad/Const*
T0*

Tmultiples0*
_output_shapes

:d
�
0gradients/loss/Mul_grad/BroadcastGradientArgs/s0Const*
valueB"d      *
dtype0*
_output_shapes
:
s
0gradients/loss/Mul_grad/BroadcastGradientArgs/s1Const*
valueB *
dtype0*
_output_shapes
: 
�
-gradients/loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients/loss/Mul_grad/BroadcastGradientArgs/s00gradients/loss/Mul_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:���������:���������*
T0
v
gradients/loss/Mul_grad/MulMulgradients/loss/Sum_grad/Tileloss/Cast/x*
T0*
_output_shapes

:d
�
gradients/loss/Mul_grad/Mul_1Mulloss/SquaredDifferencegradients/loss/Sum_grad/Tile*
_output_shapes

:d*
T0
~
-gradients/loss/Mul_grad/Sum/reduction_indicesConst*
dtype0*
valueB"       *
_output_shapes
:
�
gradients/loss/Mul_grad/SumSumgradients/loss/Mul_grad/Mul_1-gradients/loss/Mul_grad/Sum/reduction_indices*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
h
%gradients/loss/Mul_grad/Reshape/shapeConst*
_output_shapes
: *
valueB *
dtype0
j
'gradients/loss/Mul_grad/Reshape/shape_1Const*
valueB *
_output_shapes
: *
dtype0
�
gradients/loss/Mul_grad/ReshapeReshapegradients/loss/Mul_grad/Sum'gradients/loss/Mul_grad/Reshape/shape_1*
T0*
Tshape0*
_output_shapes
: 
p
(gradients/loss/Mul_grad/tuple/group_depsNoOp^gradients/loss/Mul_grad/Mul ^gradients/loss/Mul_grad/Reshape
�
0gradients/loss/Mul_grad/tuple/control_dependencyIdentitygradients/loss/Mul_grad/Mul)^gradients/loss/Mul_grad/tuple/group_deps*
_output_shapes

:d*
T0*.
_class$
" loc:@gradients/loss/Mul_grad/Mul
�
2gradients/loss/Mul_grad/tuple/control_dependency_1Identitygradients/loss/Mul_grad/Reshape)^gradients/loss/Mul_grad/tuple/group_deps*
_output_shapes
: *2
_class(
&$loc:@gradients/loss/Mul_grad/Reshape*
T0
�
,gradients/loss/SquaredDifference_grad/scalarConst1^gradients/loss/Mul_grad/tuple/control_dependency*
_output_shapes
: *
dtype0*
valueB
 *   @
�
)gradients/loss/SquaredDifference_grad/MulMul,gradients/loss/SquaredDifference_grad/scalar0gradients/loss/Mul_grad/tuple/control_dependency*
T0*
_output_shapes

:d
�
)gradients/loss/SquaredDifference_grad/subSubNet/output_layer/BiasAddInputs/y1^gradients/loss/Mul_grad/tuple/control_dependency*
_output_shapes

:d*
T0
�
+gradients/loss/SquaredDifference_grad/mul_1Mul)gradients/loss/SquaredDifference_grad/Mul)gradients/loss/SquaredDifference_grad/sub*
T0*
_output_shapes

:d
�
)gradients/loss/SquaredDifference_grad/NegNeg+gradients/loss/SquaredDifference_grad/mul_1*
T0*
_output_shapes

:d
�
6gradients/loss/SquaredDifference_grad/tuple/group_depsNoOp*^gradients/loss/SquaredDifference_grad/Neg,^gradients/loss/SquaredDifference_grad/mul_1
�
>gradients/loss/SquaredDifference_grad/tuple/control_dependencyIdentity+gradients/loss/SquaredDifference_grad/mul_17^gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*
_output_shapes

:d*>
_class4
20loc:@gradients/loss/SquaredDifference_grad/mul_1
�
@gradients/loss/SquaredDifference_grad/tuple/control_dependency_1Identity)gradients/loss/SquaredDifference_grad/Neg7^gradients/loss/SquaredDifference_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/loss/SquaredDifference_grad/Neg*
_output_shapes

:d
�
3gradients/Net/output_layer/BiasAdd_grad/BiasAddGradBiasAddGrad>gradients/loss/SquaredDifference_grad/tuple/control_dependency*
T0*
_output_shapes
:*
data_formatNHWC
�
8gradients/Net/output_layer/BiasAdd_grad/tuple/group_depsNoOp4^gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad?^gradients/loss/SquaredDifference_grad/tuple/control_dependency
�
@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependencyIdentity>gradients/loss/SquaredDifference_grad/tuple/control_dependency9^gradients/Net/output_layer/BiasAdd_grad/tuple/group_deps*>
_class4
20loc:@gradients/loss/SquaredDifference_grad/mul_1*
_output_shapes

:d*
T0
�
Bgradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad9^gradients/Net/output_layer/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*F
_class<
:8loc:@gradients/Net/output_layer/BiasAdd_grad/BiasAddGrad
�
-gradients/Net/output_layer/MatMul_grad/MatMulMatMul@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency&Net/output_layer/MatMul/ReadVariableOp*
transpose_a( *
transpose_b(*
_output_shapes

:d
*
T0
�
/gradients/Net/output_layer/MatMul_grad/MatMul_1MatMulNet/hidden_layer/Relu@gradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes

:
*
T0
�
7gradients/Net/output_layer/MatMul_grad/tuple/group_depsNoOp.^gradients/Net/output_layer/MatMul_grad/MatMul0^gradients/Net/output_layer/MatMul_grad/MatMul_1
�
?gradients/Net/output_layer/MatMul_grad/tuple/control_dependencyIdentity-gradients/Net/output_layer/MatMul_grad/MatMul8^gradients/Net/output_layer/MatMul_grad/tuple/group_deps*
_output_shapes

:d
*@
_class6
42loc:@gradients/Net/output_layer/MatMul_grad/MatMul*
T0
�
Agradients/Net/output_layer/MatMul_grad/tuple/control_dependency_1Identity/gradients/Net/output_layer/MatMul_grad/MatMul_18^gradients/Net/output_layer/MatMul_grad/tuple/group_deps*
_output_shapes

:
*B
_class8
64loc:@gradients/Net/output_layer/MatMul_grad/MatMul_1*
T0
�
-gradients/Net/hidden_layer/Relu_grad/ReluGradReluGrad?gradients/Net/output_layer/MatMul_grad/tuple/control_dependencyNet/hidden_layer/Relu*
_output_shapes

:d
*
T0
�
3gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/Net/hidden_layer/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:

�
8gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_depsNoOp4^gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad.^gradients/Net/hidden_layer/Relu_grad/ReluGrad
�
@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/Net/hidden_layer/Relu_grad/ReluGrad9^gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes

:d
*@
_class6
42loc:@gradients/Net/hidden_layer/Relu_grad/ReluGrad
�
Bgradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad9^gradients/Net/hidden_layer/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:
*F
_class<
:8loc:@gradients/Net/hidden_layer/BiasAdd_grad/BiasAddGrad
�
-gradients/Net/hidden_layer/MatMul_grad/MatMulMatMul@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency&Net/hidden_layer/MatMul/ReadVariableOp*
transpose_b(*
transpose_a( *
T0*
_output_shapes

:d
�
/gradients/Net/hidden_layer/MatMul_grad/MatMul_1MatMulInputs/x@gradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:
*
transpose_a(*
T0
�
7gradients/Net/hidden_layer/MatMul_grad/tuple/group_depsNoOp.^gradients/Net/hidden_layer/MatMul_grad/MatMul0^gradients/Net/hidden_layer/MatMul_grad/MatMul_1
�
?gradients/Net/hidden_layer/MatMul_grad/tuple/control_dependencyIdentity-gradients/Net/hidden_layer/MatMul_grad/MatMul8^gradients/Net/hidden_layer/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Net/hidden_layer/MatMul_grad/MatMul*
_output_shapes

:d
�
Agradients/Net/hidden_layer/MatMul_grad/tuple/control_dependency_1Identity/gradients/Net/hidden_layer/MatMul_grad/MatMul_18^gradients/Net/hidden_layer/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:
*B
_class8
64loc:@gradients/Net/hidden_layer/MatMul_grad/MatMul_1
b
GradientDescent/learning_rateConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
�
KGradientDescent/update_Net/hidden_layer/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentNet/hidden_layer/kernelGradientDescent/learning_rateAgradients/Net/hidden_layer/MatMul_grad/tuple/control_dependency_1*
T0**
_class 
loc:@Net/hidden_layer/kernel*
use_locking( 
�
IGradientDescent/update_Net/hidden_layer/bias/ResourceApplyGradientDescentResourceApplyGradientDescentNet/hidden_layer/biasGradientDescent/learning_rateBgradients/Net/hidden_layer/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@Net/hidden_layer/bias
�
KGradientDescent/update_Net/output_layer/kernel/ResourceApplyGradientDescentResourceApplyGradientDescentNet/output_layer/kernelGradientDescent/learning_rateAgradients/Net/output_layer/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Net/output_layer/kernel
�
IGradientDescent/update_Net/output_layer/bias/ResourceApplyGradientDescentResourceApplyGradientDescentNet/output_layer/biasGradientDescent/learning_rateBgradients/Net/output_layer/BiasAdd_grad/tuple/control_dependency_1*
T0*(
_class
loc:@Net/output_layer/bias*
use_locking( 
�
GradientDescentNoOpJ^GradientDescent/update_Net/hidden_layer/bias/ResourceApplyGradientDescentL^GradientDescent/update_Net/hidden_layer/kernel/ResourceApplyGradientDescentJ^GradientDescent/update_Net/output_layer/bias/ResourceApplyGradientDescentL^GradientDescent/update_Net/output_layer/kernel/ResourceApplyGradientDescent
R
loss_1/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_1
Q
loss_1ScalarSummaryloss_1/tags
loss/value*
_output_shapes
: *
T0
�
initNoOp^Net/hidden_layer/bias/Assign^Net/hidden_layer/kernel/Assign^Net/output_layer/bias/Assign^Net/output_layer/kernel/Assign"�	"2
	summaries%
#
Net/h_out:0

Net/pred:0
loss_1:0"�
	variables��
�
Net/hidden_layer/kernel:0Net/hidden_layer/kernel/Assign-Net/hidden_layer/kernel/Read/ReadVariableOp:0(2>Net/hidden_layer/kernel/Initializer/stateless_random_uniform:08
�
Net/hidden_layer/bias:0Net/hidden_layer/bias/Assign+Net/hidden_layer/bias/Read/ReadVariableOp:0(2)Net/hidden_layer/bias/Initializer/zeros:08
�
Net/output_layer/kernel:0Net/output_layer/kernel/Assign-Net/output_layer/kernel/Read/ReadVariableOp:0(2>Net/output_layer/kernel/Initializer/stateless_random_uniform:08
�
Net/output_layer/bias:0Net/output_layer/bias/Assign+Net/output_layer/bias/Read/ReadVariableOp:0(2)Net/output_layer/bias/Initializer/zeros:08"
losses

loss/value:0"�
trainable_variables��
�
Net/hidden_layer/kernel:0Net/hidden_layer/kernel/Assign-Net/hidden_layer/kernel/Read/ReadVariableOp:0(2>Net/hidden_layer/kernel/Initializer/stateless_random_uniform:08
�
Net/hidden_layer/bias:0Net/hidden_layer/bias/Assign+Net/hidden_layer/bias/Read/ReadVariableOp:0(2)Net/hidden_layer/bias/Initializer/zeros:08
�
Net/output_layer/kernel:0Net/output_layer/kernel/Assign-Net/output_layer/kernel/Read/ReadVariableOp:0(2>Net/output_layer/kernel/Initializer/stateless_random_uniform:08
�
Net/output_layer/bias:0Net/output_layer/bias/Assign+Net/output_layer/bias/Read/ReadVariableOp:0(2)Net/output_layer/bias/Initializer/zeros:08"
train_op

GradientDescent�%d�	      ��t	կRI��A*�
�	
	Net/h_out*�	   ����?     @�@! �W���X@)=p����?@2�        �-���q=��bȬ�0?��82?
����G?�qU���I?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             @@              �?              �?              �?              �?              �?              �?              �?      �?       @       @      �?      �?      @       @      @      �?       @       @      �?      @      @      @      @      @      @      @      @      @      @       @      @      @      @      @      @      @      @      @      @      &@      "@      "@      &@      &@      *@      "@      *@      $@      (@      ,@      1@      2@      2@      4@      8@      7@      >@      :@      7@      5@      1@      @      @        
�	
Net/pred*�		   @)ſ   �W�?      Y@!  V�,Y)@)���� @2��QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� ��&b՞
�u�hyO�s��m9�H�[���bB�SY�o��5sz?���T}?�Rc�ݒ?^�S���?�/��?�uS��a�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �?      @      @      @      @      @      @      @       @       @       @       @      �?       @      �?      �?      �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?       @      �?       @       @       @       @       @      @      @      @      @      @      @      @        

loss_1��/>?%�f+      ;�E4	�RI��A*�
�	
	Net/h_out*�	   �°�?     @�@!  �-w�W@)���>@2�        �-���q=�!�A?�T���C?�qU���I?IcD���L?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �~@              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?      �?      �?      @      @       @       @      �?      �?       @       @      @       @      @      @      @      @       @      @      @      "@       @       @      &@      "@      @      @      @      "@      @      $@      $@      $@      &@      *@      "@      "@      &@      (@      (@      .@      .@      4@      0@      6@      7@      9@      7@      7@      8@      3@      .@      @      @        
�
Net/pred*�	    Y�?   @�3�?      Y@!   �gCD@)���+5@2��K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              =@      8@       @       @      @       @      @      @      @      @      @      @      @      @      @        

loss_1���=�;      g��	��RI��A*�
�
	Net/h_out*�   �i�?     @�@! �X��V@)M���9@2�        �-���q=x?�x�?��d�r?
����G?�qU���I?ܗ�SsW?��bB�SY?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             p~@              �?               @              �?              �?              �?              @               @      �?       @       @       @      �?       @      @      �?      �?      @       @      @       @      @      @      @      @      @      @      "@      @      "@      &@      $@       @      @      "@       @      @       @      $@      (@      (@      (@      *@      1@      1@      4@      4@      5@      2@      4@      4@      8@      9@      3@      5@      8@      4@      @      @        
�
Net/pred*�	   �l�?    �:�?      Y@!  �*O�<@)��(2��$@2�!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @      @      @      @      @      &@      ;@       @       @       @      @      @      @      @      @      @      @      @        

loss_1/(�=�~S�K      :�o	�RI��A*�
�	
	Net/h_out*�	   `}��?     @�@!  �?`HW@)8=4M��:@2�        �-���q=�7Kaa+?��VlQ.?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             �}@              �?              �?              �?      �?              �?              �?              �?              �?               @      �?      �?               @       @      �?      �?       @      �?      @       @       @      @      @       @      @      @      @      @      @      "@      @      @      &@      "@      $@      $@      $@      "@      @      "@      "@      &@      "@      (@      *@      0@      .@      2@      3@      5@      8@      5@      3@      6@      5@      4@      2@      6@      9@      4@       @      @        
�
Net/pred*�	   ��U�?    ��?      Y@!  �9��A@)�_�я-@2�yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              @      @      @      @      @      "@      (@      .@      $@      @      @      @      @      @      @      @        

loss_1��]=J��n�      _�9u	��RI��A*�
�
	Net/h_out*�   ఍�?     @�@! �
)e�W@)�jb��:@2�        �-���q=��%�V6?uܬ�@8?�lDZrS?<DKc��T?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             p}@              �?              �?               @              �?               @              �?              �?              �?       @      @       @      �?      �?      @      �?      @       @      @      @      @      @      @      @      @      @      @      @      (@       @      (@      (@       @       @      "@       @      @      *@      (@      *@      .@      1@      2@      4@      2@      8@      9@      <@      8@      3@      4@      2@      5@      9@      1@      &@      @        
�
Net/pred*�	   � ��?    A3�?      Y@!   z�b@@)t��SP)@2�Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @      @      @      @      @      @      @      "@      $@      $@      &@      @      @      @      @      @      @        

loss_1( D=&��      _��	�RI��A*�
�
	Net/h_out*�   @���?     @�@!  ����W@)���Q@{;@2�        �-���q=��%>��:?d�\D�X=?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?�l�P�`?���%��b?P}���h?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             0~@              �?              �?              �?               @               @               @              @       @              �?      �?      @      �?      �?      @       @      @      @      @      @      @      @      @      @      @      @      $@      @      "@      &@      $@      *@      @      "@      $@      "@       @      &@      ,@      ,@      .@      1@      2@      3@      5@      9@      9@      :@      5@      1@      3@      5@      8@      .@      ,@      @        
�
Net/pred*�	    ?��?   @���?      Y@!  ���VB@)�Y$Ox.@2��?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�               @      @      @      @      @      @      @      @      "@      $@      $@      ,@      @      @      @      @        

loss_1ͳ2=-�69+      ;�E4	\�RI��A*�
�
	Net/h_out*�   ��d�?     @�@! �0�DX@)^��-<@2�        �-���q=�!�A?�T���C?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             �}@              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?      �?       @      @       @      �?       @       @      �?      @      @      @      @      @      @      @       @      @      @      "@      "@      "@      &@      *@       @      "@      @      &@      &@      $@      ,@      (@      ,@      3@      1@      5@      5@      8@      8@      :@      =@      2@      4@      5@      7@      ,@      *@      @        
�
Net/pred*�	    �?    ��?      Y@!  P���@@)��g)��)@2���(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @       @      @      @      @      @      @      @      @      @      @      "@       @      $@      &@      @      @      @       @        

loss_13%=���K      :�o	RI��A*�
�	
	Net/h_out*�	   �q��?     @�@!  /�b_X@),-k�� =@2�        �-���q=�u�w74?��%�V6?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             ~@              �?              �?      �?      �?              �?               @               @              �?      �?              �?              �?      �?      �?      �?      �?       @       @              @      @       @      �?      @       @      @      @       @      @      @      @      $@      @      @      &@      @      (@      $@      (@      @      "@      "@      &@      $@      *@      (@      .@      0@      2@      4@      3@      5@      8@      :@      ;@      3@      4@      4@      7@      0@      ,@      "@        
�
Net/pred*�	    ���?   @���?      Y@!  ��}dB@)�>��0/@2�Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @      @      @      @      @      @      @      @      @      @      "@       @      $@      &@      @      @      @        

loss_1�=)<�+      ;�E4	DRI��A*�
�
	Net/h_out*�   ����?     @�@! �Xc��X@)W�4zm�=@2�        �-���q=��%�V6?uܬ�@8?��%>��:?<DKc��T?ܗ�SsW?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�             P~@              �?      �?              �?              �?      �?              �?      �?              �?              �?      �?      �?      @       @       @              @      �?       @      @       @      @      @      @      @       @      &@      @      @       @      $@      @      ,@      "@      (@      @      @      &@      (@      $@      &@      (@      .@      2@      0@      5@      2@      5@      7@      9@      ;@      :@      3@      5@      7@      .@      *@      &@        
�
Net/pred*�	   �Q4�?   ���?      Y@!  ���@@)"�ӟ�!+@2�8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @       @      @       @      @       @      @      @      @      @      @      @      @      @       @       @      "@      $@      @      @       @        

loss_1�T=��ě      d��	�RI��A	*�
�	
	Net/h_out*�	    hR�?     @�@!  �̚�X@)��.ܦn>@2�        �-���q=f�ʜ�7
?>h�'�?��bȬ�0?��82?a�$��{E?
����G?�qU���I?nK���LQ?�lDZrS?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             P~@              �?              �?              �?      �?              �?              �?              �?               @              �?               @      �?      �?              �?              @      �?      �?      @      @      @       @      @      @      @      @      @      @       @      @       @      @      @      &@       @      &@      &@      *@      @       @      $@      $@      "@      *@      *@      ,@      0@      1@      1@      2@      5@      5@      :@      ;@      9@      4@      3@      6@      3@      ,@      &@      �?        
�
Net/pred*�	    ���?   ���?      Y@!   ib;B@)��<h=|/@2�%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�               @      @      �?      @      @       @      @      @      @      @      @      @      @      @       @      "@      "@      &@      @      @        

loss_1N,=��5wk      :�)�	�RI��A
*�
�
	Net/h_out*�   �>l�?     @�@!  ���1Y@)�E\)}�>@2�        �-���q=��%�V6?uܬ�@8?�qU���I?IcD���L?k�1^�sO?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �~@              �?              �?      �?              �?              �?              �?              �?              �?               @              @       @      �?              @      �?       @      @       @      @      @      @      @      @      $@      @      @      "@      @       @      &@      &@      (@      "@      @      $@      &@      "@      *@      *@      (@      0@      1@      1@      2@      4@      6@      8@      ;@      9@      9@      3@      6@      3@      *@      (@      �?        
�
Net/pred*�	   `�t�?   `�T�?      Y@!  ��.!A@)P����,@2��{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @      @      �?       @       @      @       @      @       @      @      @      @      @      @      @      @      @      @       @      $@      $@      @      @        

loss_1Tq�<k�g[      f耂	�*RI��A*�
�
	Net/h_out*�   �+��?     @�@! �g_�kY@)o��I��?@2�        �-���q=uܬ�@8?��%>��:?<DKc��T?ܗ�SsW?��bB�SY?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �~@              �?              �?      �?              �?              �?               @              @      �?      �?               @      �?       @      �?      @       @      �?       @      @       @      @      @      @      @      @      @       @      @      "@      "@      @      (@      $@      &@       @       @      &@      "@      "@      *@      &@      .@      ,@      .@      .@      4@      3@      5@      9@      8@      :@      :@      4@      6@      3@      ,@      *@       @        
�
Net/pred*�	    ɾ�?   @�;�?      Y@!   ,a	B@)j��g�/@2�� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              @      �?       @      @       @       @      @      @       @      @      @      @      @      @      @      @      @       @       @      $@       @      @      �?        

loss_1{��<"5n�      8��	6RI��A*�
�
	Net/h_out*�   @Z��?     @�@! ��x0�Y@)0i�̨,@@2�        �-���q=uܬ�@8?��%>��:?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �~@              �?              �?      �?              �?              �?               @      �?      �?      �?              �?      �?               @      @      �?      �?      �?      @      �?      @       @      @      @      @      @      @      @      @      @      &@      @      (@      "@      *@       @       @      $@      "@      $@      &@      *@      ,@      *@      *@      0@      4@      3@      6@      7@      8@      9@      ;@      8@      5@      4@      *@      ,@       @        
�
Net/pred*�	    ��?   `;��?      Y@!  j�;A@)��z:��-@2�I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              @       @               @      �?       @       @       @      �?      @       @      @      @       @      @      @      @      @      @      @      @      @       @      $@      @      @        

loss_1 �<Y�7Y�      8��	f?RI��A*�
�
	Net/h_out*�    ��?     @�@!  0�m�Y@)����V�@@2�        �-���q=I�I�)�(?�7Kaa+?��VlQ.?k�1^�sO?nK���LQ?�l�P�`?���%��b?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �~@              �?      �?              �?               @              �?      �?      �?      �?               @      �?      �?       @       @       @      �?      @      �?       @      @       @      @      @      @      @      @      @      "@      @      @       @      $@      "@      "@      (@      &@       @      "@      "@      $@      &@      *@      *@      *@      *@      0@      1@      3@      5@      8@      :@      9@      ;@      4@      7@      6@      (@      .@      @        
�
Net/pred*�	   �e�?   ����?      Y@!  �M��A@){���!0@2�I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�               @      �?       @      �?       @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @       @      $@      @       @        

loss_1��< Я)�      9�߂	#JRI��A*�
�
	Net/h_out*�   `�/�?     @�@!  T4�Z@)�ur5��@@2�        �-���q=��VlQ.?��bȬ�0?<DKc��T?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�              @              �?              �?      �?              �?       @              �?              �?       @              �?      �?       @      �?      �?       @      @       @       @       @      @      @      @      @      @      @      @       @      @      @      "@       @      "@      (@      $@      &@      @       @      $@      (@      &@      &@      &@      *@      *@      0@      2@      2@      5@      7@      9@      :@      :@      8@      5@      7@      *@      *@      @        
�
Net/pred*�	   �� �?   ���?      Y@!  ��([A@)ȵP�/@2�`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              �?       @      �?      �?      �?       @               @      �?       @              @       @       @       @      @       @      @      @      @      @      @      @      @      @      @       @       @      $@      @      �?        

loss_1#�<�(D+      =��j	=TRI��A*�
�	
	Net/h_out*�	   ��a�?     @�@! �r��FZ@)�ͣ��A@2�        �-���q=I�I�)�(?�7Kaa+?d�\D�X=?���#@?k�1^�sO?nK���LQ?�lDZrS?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             0@              �?              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?               @      �?      �?       @       @      @      @      �?      �?      @       @      @      @      @      @      @      @      "@      @      @      (@      @      &@      $@      (@       @       @      "@      $@      $@      &@      (@      (@      ,@      .@      3@      1@      5@      6@      ;@      8@      8@      <@      4@      6@      .@      *@      @        
�
Net/pred*�	   ����?   ���?      Y@!   �A@)�Ɔ��^0@2��uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              �?      �?      �?      �?       @               @              �?       @       @      �?       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @       @       @       @      @      @        

loss_1�Q�<q|iqK      <�7	_RI��A*�
�	
	Net/h_out*�   `:z�?     @�@!  ���sZ@)�g�[2ZA@2�        �-���q=�vV�R9?��ڋ?��%>��:?d�\D�X=?�lDZrS?<DKc��T?��bB�SY?�m9�H�[?�l�P�`?���%��b?Tw��Nof?P}���h?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             P@              �?              �?              �?              �?              �?               @              �?              �?               @       @      �?      �?       @      @       @       @      �?      @      �?      @      @      @      @      @      @      @      @      @       @      $@      "@      "@      &@      &@      "@       @       @      "@      $@      &@      $@      *@      .@      .@      1@      3@      4@      6@      9@      9@      9@      <@      5@      7@      ,@      ,@      @        
�
Net/pred*�	    9��?   ���?      Y@!  �fiA@)�AS�t0@2�}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�               @              �?      �?               @              �?      �?      �?               @               @      �?       @       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      "@      @       @        

loss_1�r�<E{0�{      `"��	-kRI��A*�
�	
	Net/h_out*�	    ͤ�?     @�@! �{�Z@)`0!�A@2�        �-���q=�S�F !?�[^:��"?�u�w74?��%�V6?a�$��{E?
����G?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             P@              �?              �?              �?              �?              �?               @              �?               @              �?      �?              �?      �?      @      �?      �?       @       @      @      @      �?      @      @      @      @      @      @      @      @      @      @      @      $@       @      "@      ,@      "@      "@       @      @      $@      $@      $@      $@      (@      ,@      1@      1@      1@      5@      7@      7@      9@      8@      =@      7@      6@      .@      *@       @        
�
Net/pred*�	    �X�?   @ɍ�?      Y@!   ;��A@)�^��Ʃ0@2�^�S���?�"�uԖ?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�               @               @               @              �?              �?      �?               @      �?      �?      �?       @       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      "@       @      @        

loss_1�n�<�z��      ?8O\	0uRI��A*�
�	
	Net/h_out*�	    u��?     @�@!  ���Z@)Eqp-��A@2�        �-���q=�7Kaa+?��VlQ.?��%>��:?d�\D�X=?�T���C?a�$��{E?IcD���L?k�1^�sO?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �@              �?              �?              �?              �?              �?              �?               @      �?              �?               @      �?       @      �?               @      @      @      @       @       @      @       @      @      @      @      @      @       @      @      @      "@      @      $@      *@      $@       @       @      @      "@      "@      &@      $@      *@      *@      1@      .@      3@      3@      9@      6@      9@      8@      <@      8@      7@      0@      (@      "@        
�
Net/pred*�	    ��?   �2��?      Y@!  u�jA@) c���y0@2����T}?>	� �?-Ա�L�?eiS�m�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              �?              �?              �?      �?              �?              �?              �?               @              �?      �?      �?               @               @       @      �?       @       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @       @      @        

loss_1V4�<~�}m�      c0(M	��RI��A*�
�	
	Net/h_out*�	    4��?     @�@!  ���Z@)�>���B@2�        �-���q=f�ʜ�7
?>h�'�?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?�m9�H�[?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �@              �?              �?              �?              �?              �?      �?              �?               @      �?      �?              �?      �?      �?              �?      @               @      @      @       @      �?      @      �?      @       @      @      @      @      @      "@      @      @      $@      $@      &@      (@      @      @      @      "@       @      $@      (@      *@      *@      1@      .@      2@      4@      7@      7@      8@      9@      9@      ;@      5@      2@      *@      "@        
�
Net/pred*�	    tQq?   @��?      Y@!  �ϡA@)��JQ�0@2�;8�clp?uWy��r?>	� �?����=��?-Ա�L�?eiS�m�?�#�h/�?���&�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?               @      �?      �?       @      �?       @       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @      "@      @        

loss_1���<4EKI�      ?8O\	��RI��A*�
�
	Net/h_out*�   �v��?     @�@! ����[@)��h�LB@2�        �-���q=�T7��?�vV�R9?
����G?�qU���I?<DKc��T?ܗ�SsW?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�             �@              �?              �?              �?              �?              @              �?              @      �?      �?               @       @      @      @      �?      @      �?      @       @      @      @      @      @      @      @      @      @       @      @      &@      $@      &@      "@      @       @      @      "@      $@      &@      .@      &@      0@      0@      2@      4@      7@      7@      7@      8@      ;@      <@      5@      1@      ,@      "@        
�
Net/pred*�	     �v�    v�?      Y@!  ��iA@)���ȅ�0@2�*QH�x�&b՞
�u�E��{��^��m9�H�[���d�r?�5�i}1?&b՞
�u?*QH�x?>	� �?����=��?�#�h/�?���&�?�Rc�ݒ?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      "@      @        

loss_1�<��]��      ?8O\	,�RI��A*�
�
	Net/h_out*�   ���?     @�@!  x%�5[@)N���B@2�        �-���q=���#@?�!�A?�T���C?nK���LQ?�lDZrS?���%��b?5Ucv0ed?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?      �?               @              �?               @               @       @               @              �?              @       @      �?      @      @      �?      @      @      �?      @      @      @      @      @      @       @      @       @      "@      "@      (@      "@      "@      @       @       @      "@      $@      &@      *@      *@      *@      2@      3@      2@      6@      8@      8@      8@      :@      <@      5@      2@      *@      $@      �?        
�
Net/pred*�	    F$��   �)a�?      Y@!  ��A@)���eQ1@2����J�\������=���hyO�s�uWy��r�;8�clp�ߤ�(g%k?�N�W�m?;8�clp?uWy��r?#�+(�ŉ?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?               @               @              �?       @               @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @       @      @      �?        

loss_1���<�V+��      cJ	0�RI��A*�
�
	Net/h_out*�   ``6�?     @�@!  ��2R[@)@��k��B@2�        �-���q=�T���C?a�$��{E?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?              �?              �?               @              �?              �?               @               @       @      �?      �?               @      @      @       @      @       @      @      @       @      @      @      @      @      @       @      @      @      $@      "@      $@      $@      $@      @       @       @      &@      "@      &@      (@      *@      *@      2@      2@      4@      5@      8@      8@      7@      ;@      <@      6@      2@      *@      $@      �?        
�
Net/pred*�	    �ّ�   �l�?      Y@!  `�DfA@)+�!}�81@2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�&b՞
�u�hyO�s�ߤ�(g%k?�N�W�m?&b՞
�u?*QH�x?�7c_XY�?�#�h/�?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?              �?      �?               @              �?              �?              �?              �?              �?               @              �?      �?      �?               @               @      �?       @       @       @      �?       @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1m�<)%b^K      �m3�	��RI��A*�
�	
	Net/h_out*�	   ��U�?     @�@!  ��q[@)�ʬ�B@2�        �-���q=��82?�u�w74?uܬ�@8?��%>��:?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?              �?              �?              �?               @      �?              @              �?               @              @       @      @      �?      @      @      @       @       @      @      @      @      @      @      "@      @      @      &@      @      $@      $@      "@      @      @       @      &@      "@      &@      (@      *@      ,@      1@      2@      3@      5@      7@      8@      7@      <@      :@      8@      3@      (@      &@      �?        
�
Net/pred*�	    0ɔ�    _��?      Y@!  P+�A@)I�h�Р1@2��"�uԖ�^�S�����Rc�ݒ����&���#�h/�����J�\������=������T}�o��5sz�I��P=�>��Zr[v�>�N�W�m?;8�clp?-Ա�L�?eiS�m�?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?               @      �?      �?      �?       @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      "@      @      �?        

loss_1
�<�/]C      ͡W;	��RI��A*�
�
	Net/h_out*�    !j�?     @�@! �d�{�[@)@�/GC@2�        �-���q=a�Ϭ(�>8K�ߝ�>�lDZrS?<DKc��T?ܗ�SsW?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?               @      �?              �?              �?       @              �?      �?               @              �?      �?      @      @       @       @      �?      @       @      @      @      @      @      @      @      @      "@      @      @      "@      @      &@      "@      $@      @      @       @      $@      $@      &@      &@      *@      ,@      1@      0@      5@      5@      7@      6@      9@      :@      <@      9@      2@      *@      &@      �?        
�
Net/pred*�	   �Ȅ��   @h��?      Y@!  �k&cA@)2��Q�1@2��v��ab����<�A���}Y�4j���"�uԖ��Rc�ݒ����&���#�h/���7c_XY������=���>	� ���N�W�m�ߤ�(g%k��lDZrS?<DKc��T?�7c_XY�?�#�h/�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?               @      �?      �?      �?       @               @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @       @      @      @      �?        

loss_1�Y�<��q;      ��	d�RI��A*�
�
	Net/h_out*�    ���?     @�@! ��ڣ[@)�0E35C@2�        �-���q=��%�V6?uܬ�@8?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?              �?               @              �?              @      �?               @      �?      �?      @      @      �?       @      �?      @      @       @      @      @      @      @      @      @      @       @       @      @      &@      "@      "@      @      @       @      $@      $@      &@      &@      (@      .@      0@      1@      3@      6@      6@      7@      9@      9@      =@      6@      5@      ,@      $@       @        
�
Net/pred*�	   ����    ��?      Y@!  `p��A@)JKQS`�1@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�������&���#�h/��#�+(�ŉ�eiS�m��&b՞
�u�hyO�s�5Ucv0ed����%��b�-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @               @       @       @      �?      @       @      @      @      @      @      @      @      @      @      @      @      @       @      @       @        

loss_1Q�<���	      ͡W;	��RI��A*�
�
	Net/h_out*�   ���?     @�@! ���6�[@)�_b��RC@2�        �-���q=�qU���I?IcD���L?ܗ�SsW?��bB�SY?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             (�@              �?              �?              �?      �?               @              �?      �?       @      �?      �?       @      �?       @      @       @       @      �?       @      @      @      �?      @      @      @      @      @      @      @       @      @      "@      $@      $@       @      @      "@      @      &@      "@      &@      &@      ,@      ,@      .@      1@      3@      6@      6@      8@      8@      9@      <@      8@      5@      (@      (@       @        
�
Net/pred*�	   ��բ�   �U�?      Y@!  Q�^A@)�[�e�1@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ�eiS�m��-Ա�L�����J�\��P}���h?ߤ�(g%k?*QH�x?o��5sz?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @               @              �?      �?      �?      �?      �?       @      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @       @      @       @        

loss_1�<^se�      c0(M	�RI��A*�
�
	Net/h_out*�   ���?     @�@! �B��[@)n�ժtC@2�        �-���q=a�$��{E?
����G?�m9�H�[?E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             0�@              �?              �?      �?               @              @              �?      @              �?              �?      @       @       @       @       @       @      @      @       @      @      @      @      @      @      @       @      @      @      $@      "@      &@       @      @       @      @      &@      "@      &@      &@      ,@      *@      1@      0@      3@      6@      5@      8@      9@      9@      <@      8@      5@      (@      (@       @        
�
Net/pred*�	   @�ף�    s]�?      Y@!  �"A}A@)�Ԝ+2@2�`��a�8���uS��a���/����v��ab��}Y�4j���"�uԖ�^�S����#�+(�ŉ�eiS�m����%�V6��u�w74�hyO�s?&b՞
�u?�7c_XY�?�#�h/�?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @      �?      �?              �?      �?               @              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?       @               @      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @       @        

loss_1�|�<G"�      ͡W;	e�RI��A*�
�
	Net/h_out*�   `M��?     @�@! �����[@)��, �C@2�        �-���q=��%>��:?d�\D�X=?��bB�SY?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             0�@              �?               @      �?              �?       @      �?              �?              �?      �?      �?       @      �?      �?       @      �?      @       @      @       @      @      @       @      @      @      @      @      @      @      @      @      @       @       @      (@       @      @      @      "@      "@      "@      &@      (@      *@      *@      0@      1@      3@      5@      5@      9@      7@      :@      =@      8@      4@      *@      (@       @        
�
Net/pred*�	   @���   `Cf�?      Y@!  p�rZA@)!�j�2@2��g���w���/�*>��`��a�8���uS��a���v��ab����<�A���^�S�����Rc�ݒ����&��>	� �����T}��lDZrS�nK���LQ�o��5sz?���T}?���&�?�Rc�ݒ?^�S���?�"�uԖ?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      �?       @               @              �?      �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?               @      �?      �?       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @       @       @        

loss_1�J~<�g�[      �t\�	D�RI��A*�
�	
	Net/h_out*�	    o��?     @�@! @��n�[@)D+T�S�C@2�        �-���q=��82?�u�w74?�!�A?�T���C?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             8�@              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @              �?      �?      �?      @      �?       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @      @       @       @      &@      $@      @      @      "@      "@      $@      &@      &@      (@      ,@      0@      0@      2@      7@      5@      8@      8@      :@      <@      8@      5@      *@      (@       @        
�
Net/pred*�	    1v��   `n��?      Y@!  ��0wA@)*�?�i2@2��g���w���/�*>��`��a�8���/����v��ab���"�uԖ�^�S�����Rc�ݒ�eiS�m��-Ա�L�����%��b��l�P�`�;8�clp?uWy��r?���&�?�Rc�ݒ?^�S���?�/��?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @       @               @              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?              �?      �?               @      �?      �?       @      �?       @       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @      @        

loss_1�z<16�;      ��	��RI��A*�
�
	Net/h_out*�   ����?     @�@! @7�r�[@)S̪ڸC@2�        �-���q=x?�x�?��d�r?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             H�@              �?              �?              �?              �?      �?              �?              �?               @      �?       @      �?      �?              @       @       @      @      �?       @      @      @      @      @      @      @      @      @      @      @       @      @      "@       @      $@      $@      @      @      "@      $@      "@      $@      (@      (@      .@      .@      0@      2@      6@      6@      8@      8@      9@      <@      9@      5@      *@      &@      @        
�
Net/pred*�	   �!���   ����?      Y@!   �mVA@)9�$rZ2@2����g�骿�g���w���/�*>��`��a�8���uS��a���/�����<�A���}Y�4j���"�uԖ����&���#�h/������=���>	� ��5Ucv0ed����%��b�eiS�m�?#�+(�ŉ?�7c_XY�?��<�A��?�v��ab�?�/��?`��a�8�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @       @              �?      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?               @               @               @      �?       @               @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @       @      @        

loss_1Vv<p���      ͡W;	��RI��A*�
�
	Net/h_out*�   �q��?     @�@! ���[@)��u�C@2�        �-���q=<DKc��T?ܗ�SsW?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             X�@              �?              �?      �?              �?              �?              @       @      �?      �?      �?      �?       @      @      �?       @       @      @      @      @       @      @      @      @      @      @      @      @      "@      @      "@       @      (@       @      @      @      "@      $@      $@      "@      (@      (@      .@      0@      ,@      4@      6@      5@      8@      7@      :@      ;@      :@      5@      *@      &@      @        
�
Net/pred*�	   �1y��   �	��?      Y@!  @=�lA@)?J��6�2@2����g�骿�g���w���/�*>��`��a�8���uS��a���v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����J�\������=���*QH�x�&b՞
�u����J�\�?-Ա�L�?eiS�m�?}Y�4j�?��<�A��?�v��ab�?�/��?`��a�8�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @       @               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?               @               @      �?      �?      �?       @               @       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @      @      @        

loss_1$�r<\��K      �m3�	�RI��A *�
�
	Net/h_out*�   `\��?     @�@!  �uk�[@)�{���C@2�        �-���q=nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             X�@              �?              �?              �?              �?              �?       @      �?               @      �?       @              �?       @      �?       @       @      �?       @      @      @      @      �?      @      @      @      @      @      @       @      @      @      $@      @      *@       @       @      @       @      $@      $@      "@      &@      ,@      *@      0@      .@      4@      6@      5@      7@      7@      :@      ;@      :@      6@      *@      &@      @        
�
Net/pred*�	    V^��   ����?      Y@!  � �QA@)���l�2@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ��7c_XY��#�+(�ŉ�eiS�m���N�W�m?;8�clp?o��5sz?���T}?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�/��?�uS��a�?�/�*>�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @       @      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @       @      �?       @       @       @       @      @      @       @      @      @      @      @      @      @      @       @      @      @        

loss_1��o<�![      �t\�	RI��A!*�
�	
	Net/h_out*�   ����?     @�@! ����[@)>60 ��C@2�        �-���q=k�1^�sO?nK���LQ?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             X�@               @              �?      �?              �?      �?              �?              �?      �?               @      �?              �?      �?       @               @      �?       @       @      @      @      @       @      �?      @      @      @      @      @      @       @      @       @       @      @      ,@       @      @      @       @      "@      $@      "@      &@      *@      ,@      .@      0@      3@      6@      6@      7@      6@      ;@      :@      :@      7@      *@      &@      @        
�
Net/pred*�	   ����   @6�?      Y@!  �EhA@)ڕ�	��2@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���#�h/���7c_XY��<DKc��T?ܗ�SsW?*QH�x?o��5sz?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @       @      �?      �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?       @               @       @      �?       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @       @      @        

loss_1W�l<{��	      �@.	�RI��A"*�
�
	Net/h_out*�   ����?     @�@! �W���[@)J�MK�C@2�        �-���q=��%>��:?d�\D�X=?�qU���I?IcD���L?��bB�SY?�m9�H�[?�l�P�`?���%��b?5Ucv0ed?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             h�@              �?               @              �?              �?      �?              �?              @      �?              �?      �?      �?               @      @       @      @      @       @       @       @      �?      @      @      @      @      @      @       @      @       @       @      @      *@       @      @       @      @      $@      "@      $@      &@      &@      1@      ,@      .@      3@      5@      6@      7@      7@      :@      ;@      :@      7@      *@      &@      @        
�
Net/pred*�	   @=��   �]�?      Y@!  ���MA@)I�n��2@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab����<�A���^�S�����Rc�ݒ�uWy��r�;8�clp��m9�H�[?E��{��^?eiS�m�?#�+(�ŉ?^�S���?�"�uԖ?��<�A��?�v��ab�?`��a�8�?�/�*>�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      @      �?      �?      �?              �?      �?               @              �?              �?              �?              �?              �?               @               @              �?      �?      �?               @      �?      �?       @       @      �?       @       @      @       @      @       @      @      @      @      @      @      @      @      @       @      @        

loss_1��i<���      ?8O\	u%RI��A#*�
�
	Net/h_out*�   �v��?     @�@! @.S��[@)�m���C@2�        �-���q=
����G?�qU���I?�l�P�`?���%��b?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?               @              �?       @               @              �?      �?      �?      @      �?      �?      @      @       @      �?      @      �?      @      @      @      @      @      @      @      @      @       @      @      @      (@      $@      @      @       @      "@      "@      $@      &@      (@      0@      ,@      0@      2@      5@      6@      8@      6@      9@      <@      :@      7@      (@      (@      @        
�
Net/pred*�	   ��<��   ��>�?      Y@!  @�eA@)��.��2@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab��^�S�����Rc�ݒ�o��5sz�*QH�x�ܗ�SsW?��bB�SY?���J�\�?-Ա�L�?^�S���?�"�uԖ?}Y�4j�?��<�A��?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      @      �?               @               @               @              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?               @      �?      �?       @      �?       @       @       @      @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1��g<瘊.      ͡W;	�1RI��A$*�
�
	Net/h_out*�   �U��?     @�@!  ��)�[@)D��wE�C@2�        �-���q=�T���C?a�$��{E?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?               @              �?              �?               @      �?               @      �?               @       @      �?       @       @      @       @       @       @       @       @      @      @      @      @      @      @      @      @       @       @      "@      $@      $@      @      @      "@       @       @      &@      &@      (@      1@      *@      1@      1@      5@      6@      8@      5@      :@      ;@      <@      6@      (@      &@      @        
�
Net/pred*�	   �ͬ��   @�?�?      Y@!  @��IA@)Ӭ{ll�2@2�����iH��I�������g�骿�g���w���/�*>���uS��a���/���}Y�4j���"�uԖ�^�S����eiS�m��-Ա�L��Tw��Nof�5Ucv0ed�uWy��r?hyO�s?���&�?�Rc�ݒ?^�S���?�"�uԖ?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @      @               @               @              �?      �?              �?              �?              �?              �?              �?               @              �?              �?      �?              �?      �?       @               @      �?      �?       @      �?       @       @      @      @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1\e<ܬ�K      �m3�	�;RI��A%*�
�
	Net/h_out*�   @m��?     @�@!  
���[@)�mة��C@2�        �-���q=���#@?�!�A?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?      �?              �?      �?      �?              �?      �?               @      �?      �?       @      �?      �?      �?       @      @      �?      @       @       @      @      @      @      @      @      @       @       @      @      @      "@      "@      "@      $@      @       @      "@      @      "@      $@      (@      &@      1@      *@      1@      2@      4@      6@      8@      5@      9@      ;@      =@      6@      *@      $@      @        
�
Net/pred*�	   �ut��   �Ah�?      Y@!  ���aA@)H���("3@2�����iH��I�������g�骿�g���w���/�*>���uS��a���/�����<�A���}Y�4j���"�uԖ�^�S�����7c_XY��#�+(�ŉ�5Ucv0ed����%��b�P}���h?ߤ�(g%k?���&�?�Rc�ݒ?^�S���?�/��?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @      @               @               @              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?               @               @               @      �?      �?       @      �?       @       @      @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1�Pc<��[[      �t\�	�FRI��A&*�
�	
	Net/h_out*�   ����?     @�@! �(|�[@)ɂ�H=�C@2�        �-���q=��%�V6?uܬ�@8?��%>��:?d�\D�X=?�qU���I?IcD���L?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?               @              �?      �?      �?               @      �?      �?      �?               @      �?      �?              �?      @      @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      "@      @      &@      $@      @      @      $@      @      "@      $@      &@      *@      .@      ,@      0@      4@      4@      5@      7@      6@      7@      <@      =@      7@      (@      $@      @        
�
Net/pred*�	   ��Y��    �d�?      Y@!  ��|FA@)�k�203@2���]$A鱿����iH��I�������g�骿�g���w��`��a�8���uS��a���v��ab����<�A���}Y�4j���"�uԖ��Rc�ݒ����&��o��5sz�*QH�x�E��{��^��m9�H�[��7c_XY�?�#�h/�?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?       @       @       @               @              �?              �?              �?              �?              �?               @              �?      �?              �?              �?      �?              �?      �?               @               @      �?       @      �?       @      �?      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�ma<����K      �m3�	�RRI��A'*�
�
	Net/h_out*�   ����?     @�@! @�D0�[@)���f�C@2�        �-���q=��%�V6?uܬ�@8?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?      �?      �?              �?              �?              @               @      �?      �?              �?              @      @       @      �?      @      �?      @       @      @      @      @      @      @      @      @      @      @      "@      "@      "@      &@      @      @      $@       @      "@      "@      (@      (@      .@      ,@      0@      4@      4@      5@      7@      6@      7@      <@      =@      7@      (@      $@      @        
�
Net/pred*�	   �j#��   ���?      Y@!  (2K_A@)m_���E3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ�o��5sz�*QH�x�P}���h�Tw��Nof�#�+(�ŉ?�7c_XY�?�#�h/�?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�               @      @       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?               @               @               @      �?       @      �?      �?       @      @       @      @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1��_<�5|�[      �t\�	1]RI��A(*�
�
	Net/h_out*�   �S��?     @�@! @�۽[@)2D�/��C@2�        �-���q=��VlQ.?��bȬ�0?�m9�H�[?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?       @               @               @      �?      �?              �?      �?      �?               @              @       @      @       @       @       @       @      @      @      @      �?      @      @      @      @      @      @      "@       @      $@      $@      @      @       @       @      $@      $@      &@      &@      .@      .@      0@      4@      4@      4@      7@      7@      7@      ;@      >@      6@      (@      $@      @        
�
Net/pred*�	    ����   ����?      Y@!  H^�DA@)��3[03@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S����-Ա�L�����J�\��>	� �����T}�����=��?���J�\�?-Ա�L�?eiS�m�?}Y�4j�?��<�A��?�v��ab�?�/��?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      @       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @       @       @      �?       @       @      @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1}&^<��(q[      �t\�	�fRI��A)*�
�
	Net/h_out*�   �P��?     @�@! �	���[@)ʰ�8��C@2�        �-���q=�7Kaa+?��VlQ.?
����G?�qU���I?��bB�SY?�m9�H�[?E��{��^?�l�P�`?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?              @       @              �?              �?      �?      �?      �?              �?       @       @       @       @      @       @       @      @      �?      @      @      @      @      @      @      @      @      @      "@       @      $@      "@      @      @      "@      @      $@      $@      &@      &@      1@      ,@      .@      4@      4@      5@      5@      8@      7@      ;@      >@      5@      *@      &@      @        
�
Net/pred*�	   ��k��   `���?      Y@!  ��B[A@)E{�4�a3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/���}Y�4j���"�uԖ�^�S����-Ա�L�����J�\������=���>	� ��>	� �?����=��?eiS�m�?#�+(�ŉ?}Y�4j�?��<�A��?�/��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      @      �?       @      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?      �?      �?      �?      �?       @       @       @      �?       @       @      @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1��\<�U�$;      ��	$qRI��A**�
�	
	Net/h_out*�   �P��?     @�@! �p4
�[@)Nu����C@2�        �-���q=ji6�9�?�S�F !?�T���C?a�$��{E?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?               @              �?      �?              @              �?              �?      �?      �?              �?      �?      �?              �?       @      �?       @      @      @      �?      @      @      �?      @      @       @      @      @      @      @      @      @      "@       @      &@      "@      @      @       @       @      "@      &@      $@      (@      0@      ,@      0@      3@      4@      4@      6@      7@      9@      ;@      =@      5@      *@      &@      @        
�
Net/pred*�	   ��ٰ�    A��?      Y@!  ��AA@)����K3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a����<�A���}Y�4j���7c_XY��#�+(�ŉ��N�W�m?;8�clp?>	� �?����=��?^�S���?�"�uԖ?��<�A��?�v��ab�?�uS��a�?`��a�8�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      @       @      �?      �?               @               @               @              �?              �?              �?              �?              �?              �?      �?               @               @              �?       @      �?      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1X![<���nk      ̃h	�zRI��A+*�
�	
	Net/h_out*�   �B��?     @�@!  �\��[@) 9�1�C@2�        �-���q=ji6�9�?�S�F !?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?      �?      �?      �?              �?              �?               @      �?              �?              �?              @              �?      @      @       @       @      @      �?      �?      @      @      @      @      @      @      @      @      @      "@      @      $@      $@      "@      @      @       @      $@      &@      $@      (@      0@      *@      1@      2@      3@      6@      7@      6@      8@      ;@      =@      5@      *@      &@      @        
�
Net/pred*�	   �~��   ����?      Y@!  h/�YA@)����]}3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a����<�A���}Y�4j���"�uԖ��7c_XY��#�+(�ŉ�eiS�m��P}���h?ߤ�(g%k?����=��?���J�\�?�Rc�ݒ?^�S���?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @       @      �?      �?               @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @      �?      �?       @       @      �?       @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1��Y<��]{      �zny		�RI��A,*�
�	
	Net/h_out*�	    ~��?     @�@!  (�֊[@)o?�"�C@2�        �-���q=����?f�ʜ�7
?���#@?�!�A?�T���C?nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?               @              �?      �?              �?              �?              �?              �?               @      �?      �?      �?      �?      @      @       @      @      �?      �?       @      @      @       @      @      @      @      @      @       @       @       @      &@      $@       @      @      "@      @       @      (@      &@      &@      0@      *@      2@      1@      4@      6@      5@      7@      8@      ;@      >@      5@      (@      &@      @        
�
Net/pred*�	   ��$��   �3��?      Y@!  i>0A@)f�fTN3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���v��ab����<�A���^�S�����Rc�ݒ��#�h/���7c_XY���l�P�`�E��{��^�uWy��r?hyO�s?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?�v��ab�?�/��?�/�*>�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      @       @      �?               @               @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?      �?      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�X<D$`�[      �t\�	��RI��A-*�
�
	Net/h_out*�   `3��?     @�@! @�N�{[@)��6z��C@2�        �-���q=ji6�9�?�S�F !?��bȬ�0?��82?k�1^�sO?nK���LQ?��bB�SY?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?      �?              �?              �?      �?              �?               @      �?      �?      �?       @      @      @      @      �?      �?       @      @      @      @      @      @      @       @      @      @       @      @      "@      "@      &@      @      @       @      @      "@      &@      &@      &@      1@      *@      1@      2@      3@      6@      5@      7@      9@      :@      =@      6@      *@      $@      @        
�
Net/pred*�	   ��D��   ���?      Y@!  ��SfA@)�qU�3@2�����iH��I�������g�骿�g���w��`��a�8���uS��a���v��ab����<�A���}Y�4j���"�uԖ����&���#�h/��eiS�m��-Ա�L��U�4@@�$��[^:��"����T}?>	� �?�#�h/�?���&�?��<�A��?�v��ab�?�/��?�uS��a�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              @       @      @               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?               @              �?       @      �?      �?       @       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @       @      @      �?        

loss_1BW<f���[      �t\�	�RI��A.*�
�
	Net/h_out*�   ����?     @�@! @�%wd[@)9u>,�tC@2�        �-���q=O�ʗ��>>�?�s��>a�$��{E?
����G?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             Ȁ@              �?              �?              �?              �?              �?              �?      �?      �?       @              �?              @      �?       @      @       @       @      �?       @       @      @      @       @      @      @      @      @      @       @      @      "@      "@       @      (@      @      @      "@       @      @      &@      (@      (@      0@      &@      2@      2@      4@      6@      5@      6@      8@      <@      =@      6@      (@      $@      @        
�
Net/pred*�	    ���    ���?      Y@!  y�,A@)�Rp��_3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab����<�A����"�uԖ�^�S�����#�h/���7c_XY��o��5sz�*QH�x�ܗ�SsW?��bB�SY?-Ա�L�?eiS�m�?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      @      �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?      �?      �?       @               @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1��U<��ϋK      �m3�	��RI��A/*�
�
	Net/h_out*�    "��?     @�@!  ��IP[@)��m�cC@2�        �-���q=��ڋ?�.�?ܗ�SsW?��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             Ѐ@              �?              �?      �?               @              �?              �?              �?       @               @      �?      �?              @      �?      @      �?      �?      @      @      @      @       @      @      @      @      @      @       @      @       @      "@      $@      $@      "@      @      "@      "@       @      (@      $@      (@      .@      (@      2@      3@      2@      5@      7@      6@      8@      ;@      =@      6@      (@      $@      @        
�
Net/pred*�	   ���   @���?      Y@!  ��cA@)�3��*�3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ�eiS�m��-Ա�L��uWy��r�;8�clp�uWy��r?hyO�s?#�+(�ŉ?�7c_XY�?}Y�4j�?��<�A��?�v��ab�?�/��?�/�*>�?�g���w�?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      @       @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?      �?      �?      �?      �?      �?      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @       @      @      �?        

loss_1F�T<�4��{      �zny	\�RI��A0*�
�
	Net/h_out*�   @���?     @�@!  ��?[@)�k��*NC@2�        �-���q=�f����>��(���>k�1^�sO?nK���LQ?ܗ�SsW?��bB�SY?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             Ȁ@              �?               @              �?              �?      �?              �?              @              �?      �?      �?      �?      �?               @      @      �?       @      @       @       @      @      @      @      @      @      @      @      @      @      @      (@      @      *@      @      &@      @      "@       @       @      (@      &@      (@      *@      0@      ,@      3@      4@      6@      5@      7@      6@      =@      =@      5@      (@      $@      @        
�
Net/pred*�	   �����   �A��?      Y@!  oFA@)��i�G]3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ����&���#�h/��eiS�m��-Ա�L�����%��b��l�P�`�*QH�x?o��5sz?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?       @               @       @               @       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1
�S<m���[      �t\�	�RI��A1*�
�
	Net/h_out*�   �2��?     @�@! �P2,[@)��tv?C@2�        �-���q=�[^:��"?U�4@@�$?���#@?�!�A?ܗ�SsW?��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ؀@              �?              �?              �?      �?              �?      �?       @              �?              �?               @               @              @      �?      @      �?       @      @      @      �?      @      @      @      @      @      @      "@      @      @      (@      @      *@      @      &@      @       @       @      $@      $@      &@      (@      ,@      0@      ,@      4@      2@      5@      6@      7@      8@      ;@      =@      5@      (@      $@      @        
�
Net/pred*�	   �D��   ���?      Y@!  ��GpA@)Z3dl�3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab��}Y�4j���"�uԖ�^�S�����Rc�ݒ�-Ա�L�����J�\�����T}�o��5sz�P}���h?ߤ�(g%k?���J�\�?-Ա�L�?}Y�4j�?��<�A��?�v��ab�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @       @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?      �?      �?      �?      �?       @               @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_14hR<G��      �ɟV	ԾRI��A2*�
�	
	Net/h_out*�	    x�?     @�@! @^2D[@)�ް�#C@2�        �-���q=�FF�G ?��[�?
����G?�qU���I?IcD���L?k�1^�sO?��bB�SY?�m9�H�[?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ؀@              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?       @      @      �?       @      @      �?      �?      @      �?       @      @      @      @      @      @      @      @      @      @      $@      @      (@       @      "@       @      @       @      "@      $@      (@      &@      ,@      0@      *@      4@      5@      6@      4@      8@      7@      ;@      <@      5@      (@      &@       @        
�
Net/pred*�	   @�ٰ�   @K��?      Y@!  RA@)hZ�!S3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A����Rc�ݒ����&���#�h/��;8�clp��N�W�m���bB�SY?�m9�H�[?�#�h/�?���&�?�Rc�ݒ?^�S���?�uS��a�?`��a�8�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      @      �?      �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?      �?      �?      �?      �?       @      �?      �?       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_11BQ<���;      ��	��RI��A3*�
�
	Net/h_out*�   ��y�?     @�@! @�e�[@)��
 %C@2�        �-���q=�7Kaa+?��VlQ.?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?              �?              �?              �?              �?              @              �?              �?       @       @      �?       @      �?       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @       @      $@      $@      @      "@      @       @      "@      &@      &@      &@      .@      0@      ,@      3@      4@      5@      6@      7@      8@      :@      =@      4@      (@      &@       @        
�
Net/pred*�	   �����   �
8�?      Y@!  ��~A@)�Cs�4@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab���"�uԖ�^�S�����Rc�ݒ����J�\������=���>	� ��uWy��r?hyO�s?>	� �?����=��?�"�uԖ?}Y�4j�?��<�A��?`��a�8�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      @       @       @              �?      �?              �?      �?              �?      �?              �?              �?              �?      �?              �?      �?              �?      �?              �?      �?      �?      �?      �?       @               @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_1TP<���k      ̃h	0�RI��A4*�
�	
	Net/h_out*�    [�?     @�@!  ��Z@)6�y3��B@2�        �-���q=>h�'�?x?�x�?�!�A?�T���C?�qU���I?IcD���L?��bB�SY?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?      �?               @              �?      �?      �?              �?              �?      �?      �?      @      �?       @       @      �?      @      @       @      �?      @       @      @      @      @      @      @      @      @       @      "@      "@      &@      @      @      "@      @      $@      $@      $@      &@      2@      (@      2@      2@      3@      4@      7@      6@      8@      ;@      <@      6@      (@      "@       @        
�
Net/pred*�	    X���   ����?      Y@!  ��/�@@)-D�&J3@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab����<�A����Rc�ݒ����&��;8�clp��N�W�m�5Ucv0ed����%��b�-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?�v��ab�?�/��?�uS��a�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      �?      @       @      �?              �?      �?               @              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?      �?      �?       @               @       @      �?       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�O<�a�Bk      ̃h	��RI��A5*�
�	
	Net/h_out*�	    $d�?     @�@! @�l�Z@)U.*�i�B@2�        �-���q=��bȬ�0?��82?�u�w74?���#@?�!�A?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?      �?              �?              �?      �?      �?              �?              �?              �?               @              �?              �?      �?      �?       @       @      �?       @       @      @      �?      @      �?      @      @      @      @      @      @      @      @      @      "@       @      "@      &@       @      @       @       @      "@      $@      &@      (@      0@      *@      2@      2@      2@      4@      8@      6@      8@      :@      =@      5@      (@      "@       @        
�
Net/pred*�	   @,��   �VV�?      Y@!  ��O�A@)���'4@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab���"�uԖ�^�S�����Rc�ݒ�-Ա�L�����J�\�����T}�o��5sz�*QH�x?o��5sz?���T}?^�S���?�"�uԖ?}Y�4j�?`��a�8�?�/�*>�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      �?      @      �?      �?      �?              �?      �?              �?              �?              �?      �?              �?      �?               @              �?      �?      �?              �?      �?      �?      �?       @               @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_1��M<g$�q      ͡W;	��RI��A6*�
�
	Net/h_out*�   ��U�?     @�@!  ��Z@).R��B@2�        �-���q=�vV�R9?��ڋ?uܬ�@8?��%>��:?��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?               @              �?              @              �?      �?      �?      �?      @      @      �?       @      �?       @      @      @      @      @      @      @      @      @      @       @       @      @      "@      $@      $@      $@      @      @      "@      "@      &@      &@      &@      ,@      *@      1@      4@      2@      5@      5@      7@      7@      ;@      =@      5@      (@      "@       @        
�
Net/pred*�	    ���   ����?      Y@!  ��	�@@)?�Q3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���/����v��ab��^�S�����Rc�ݒ����&���#�h/��&b՞
�u�hyO�s�ߤ�(g%k�P}���h����J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�v��ab�?�/��?�/�*>�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      @       @       @       @               @              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?               @               @               @       @      �?       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1��L<��s      ͡W;	��RI��A7*�
�
	Net/h_out*�   @@k�?     @�@! �ܗ��Z@)�1��ߴB@2�        �-���q=�u�w74?��%�V6?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?              �?              �?               @              �?       @      �?               @      �?      �?      @      �?      @      �?      �?       @      @      @      @      @      @      @      @      @      @      @       @      @       @      $@      $@      $@      @      @      $@       @      &@      &@      (@      *@      ,@      1@      4@      1@      6@      5@      8@      6@      :@      =@      5@      (@      "@       @        
�
Net/pred*�	    ��   ��\�?      Y@!  Xz�A@)S+�^a/4@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab���"�uԖ�^�S�����Rc�ݒ�-Ա�L�����J�\��&b՞
�u�hyO�s��N�W�m?;8�clp?>	� �?����=��?^�S���?�"�uԖ?�uS��a�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      @      �?       @      �?       @              �?      �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?      �?      �?               @               @      �?      �?       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_1ZK<Pl���      �ɟV	��RI��A8*�
�	
	Net/h_out*�    �^�?     @�@! �z}��Z@)U�� �B@2�        �-���q=�vV�R9?��ڋ?d�\D�X=?���#@?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?      �?      �?      �?       @      �?      @               @      @      �?      @       @      @      @      @      @      @      @      @      @      @      "@      @      &@      "@      "@      @      "@      @      $@      $@      &@      *@      *@      ,@      0@      4@      2@      6@      6@      5@      8@      <@      <@      3@      (@      "@       @        
�
Net/pred*�	   �RQ��   @���?      Y@!  @�#�@@)��!7�C3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A����"�uԖ�^�S�������&���#�h/������=���>	� ��5Ucv0ed����%��b�����=��?���J�\�?-Ա�L�?eiS�m�?}Y�4j�?��<�A��?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      @      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              �?      �?      �?      �?               @      �?      �?       @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1x@J<nOk      ̃h	�RI��A9*�
�
	Net/h_out*�   �}w�?     @�@!  �DwZ@)�'�ҧ�B@2�        �-���q=O�ʗ��>>�?�s��>��%�V6?uܬ�@8?�lDZrS?<DKc��T?�m9�H�[?E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?              �?       @               @              �?      �?               @               @      �?      �?       @      �?       @      @      �?      �?      @      @      @      @      @       @      @      @       @      @      @      $@      @      &@       @      &@      @       @       @      "@      &@      &@      *@      &@      1@      .@      4@      2@      6@      6@      4@      9@      ;@      <@      3@      (@      "@       @        
�
Net/pred*�	   ��G��   ��y�?      Y@!  ���A@)��ʠ�L4@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A����"�uԖ�^�S�������&���#�h/��-Ա�L�����J�\��Tw��Nof�5Ucv0ed�5Ucv0ed?Tw��Nof?-Ա�L�?eiS�m�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @       @      �?       @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?               @               @       @               @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @       @        

loss_1	I<G~�      �ɟV	�RI��A:*�
�	
	Net/h_out*�   `g�?     @�@! ��RZ@)tX#L�]B@2�        �-���q=��ڋ?�.�?���#@?�!�A?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?              �?       @      �?       @      �?      @      �?      �?       @      @       @       @      @      @      @      @      @      @      @      @      @      "@      @      &@      "@      "@       @       @      @       @      &@      (@      &@      0@      ,@      ,@      3@      3@      6@      5@      7@      6@      <@      >@      1@      (@      $@      �?        
�
Net/pred*�	   �%��    W��?      Y@!  �,�@@)x{N^(G3@2�����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A����"�uԖ�^�S�����#�h/���7c_XY��-Ա�L�����J�\��IcD���L��qU���I�o��5sz?���T}?#�+(�ŉ?�7c_XY�?}Y�4j�?��<�A��?`��a�8�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      �?       @       @       @              �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?      �?      �?      �?      �?       @               @       @      �?       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�G<8�{      �zny	�RI��A;*�
�
	Net/h_out*�   �>��?     @�@! �S͈@Z@)j�R�rSB@2�        �-���q=��%�V6?uܬ�@8?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             �@              �?              �?              �?               @               @              �?              �?       @               @      �?      �?       @      �?      @      �?      @      @       @       @      @      @      @      @      @      @      @      @      @      "@      "@      &@       @      $@      @      @       @      @      $@      (@      *@      *@      0@      .@      2@      4@      5@      5@      6@      8@      ;@      =@      1@      (@       @      @        
�
Net/pred*�	   �D���    ���?      Y@!  Xz֛A@)D��d-Q4@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����#�h/���7c_XY��-Ա�L�����J�\���T���C��!�A�<DKc��T?ܗ�SsW?#�+(�ŉ?�7c_XY�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�/��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?      �?       @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?               @               @       @               @       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @       @        

loss_1/xF<éb[�      �ɟV	d$RI��A<*�
�
	Net/h_out*�   @*r�?     @�@!  ��%Z@)�A,1F.B@2�        �-���q=�T7��?�vV�R9?��%>��:?d�\D�X=?a�$��{E?
����G?��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�              �@              �?              �?              �?              �?               @              �?              �?      �?      �?       @              �?       @       @       @      �?      @      �?      �?      @      �?      @      @      @      @      @      @      @      @       @      @       @      $@      "@      "@       @      @       @      "@      &@      $@      *@      *@      .@      0@      2@      3@      6@      5@      7@      8@      :@      <@      2@      (@      "@      �?        
�
Net/pred*�	   �g0��   @��?      Y@!  0.A�@@)#
D�\73@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L��d�\D�X=?���#@?ߤ�(g%k?�N�W�m?�7c_XY�?�#�h/�?^�S���?�"�uԖ?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      @       @      �?      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?       @               @               @       @       @       @      �?      @       @       @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�dE<S�U5K      �m3�	�.RI��A=*�
�
	Net/h_out*�    ���?     @�@! @% aZ@)�^}ĳ'B@2�        �-���q=�.�?ji6�9�?uܬ�@8?��%>��:?E��{��^?�l�P�`?���%��b?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             (�@              �?              �?              �?      �?              �?      �?              �?      �?              @              �?      �?      �?       @       @       @      @       @      �?      @      @       @      @      @       @      @       @      @      @       @      @      @      $@      &@      $@      @      @      &@      @      &@      (@      (@      (@      0@      1@      1@      3@      5@      5@      7@      8@      ;@      ;@      2@      $@      $@       @        
�
Net/pred*�	    �:��    ���?      Y@!  ��̩A@)�$��j4@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ�eiS�m��-Ա�L�����J�\��E��{��^?�l�P�`?���%��b?5Ucv0ed?�#�h/�?���&�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?      �?       @      �?      �?      �?      �?              �?      �?              �?              �?               @              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?      �?      �?       @      �?      �?       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @       @        

loss_1�,D<��{      �zny	�8RI��A>*�
�
	Net/h_out*�   ��z�?     @�@! ��#
�Y@)G^�Rd�A@2�        �-���q=��ڋ?�.�?a�$��{E?
����G?�qU���I?��bB�SY?�m9�H�[?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             (�@              �?              �?      �?              �?              �?              �?               @              �?       @      �?              �?       @      �?      �?       @      @      �?       @      �?      @      @      @      @      @      @      @      @      @      @      @       @      "@      $@      "@      "@      @       @      $@      @      &@      (@      (@      (@      0@      1@      1@      3@      5@      5@      6@      9@      9@      ?@      0@      (@       @      �?        
�
Net/pred*�	   @(櫿   @���?      Y@!  �}�@@)3a_��=3@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�eiS�m��-Ա�L��ܗ�SsW?��bB�SY?�l�P�`?���%��b?�#�h/�?���&�?�Rc�ݒ?^�S���?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      �?       @       @       @      �?              �?      �?               @              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?               @               @      �?      �?       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1MC<.ۆ�[      �t\�	uCRI��A?*�
�
	Net/h_out*�   ����?     @�@! �5q)�Y@)�&�QF�A@2�        �-���q=uܬ�@8?��%>��:?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             0�@              �?              �?              �?      �?      �?              �?      �?               @      �?               @               @              �?      �?       @      @      �?       @       @      @      @       @      @      @      @      @      @      @      @      @      @      "@      "@      &@      "@      @       @       @       @      &@      (@      (@      (@      .@      2@      0@      4@      6@      5@      5@      8@      9@      <@      3@      &@       @       @        
�
Net/pred*�	   @���   �	��?      Y@!  �E�A@)�E��p4@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&��-Ա�L�����J�\����bB�SY?�m9�H�[?�N�W�m?;8�clp?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      @              �?       @      �?               @      �?               @              �?              �?              �?              �?               @              �?      �?              �?      �?      �?              �?      �?      �?      �?      �?       @      �?      �?       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�B<+XL�k      ̃h	TLRI��A@*�
�
	Net/h_out*�   ����?     @�@! �~��Y@)��$G�A@2�        �-���q=��ڋ?�.�?
����G?�qU���I?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             @�@              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?      �?      @      @      �?      �?      @      �?      @       @      @      @       @      @      @      @      @      @      @      &@      @      &@      "@      &@      @      @      "@       @      $@      (@      (@      *@      0@      .@      4@      3@      6@      3@      6@      9@      :@      =@      0@      (@      @       @        
�
Net/pred*�	   � B��    ��?      Y@!   �~�@@)����/3@2�I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ��7c_XY��#�+(�ŉ�eiS�m��-Ա�L��ܗ�SsW�<DKc��T�Tw��Nof?P}���h?�#�h/�?���&�?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      @      �?       @      �?       @      �?               @              �?              �?              �?              �?               @              �?      �?              �?      �?      �?              �?              �?      �?               @               @       @               @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�OA<�&�V�      ���I	WRI��AA*�
�
	Net/h_out*�   ���?     @�@! ��?�Y@)�}�-3�A@2�        �-���q=uܬ�@8?��%>��:?d�\D�X=?���#@?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             @�@              �?              �?              �?              �?              �?      �?      �?               @              �?              �?               @      @       @      @       @       @       @       @       @      @      @      @      @      @       @      @      @       @      @       @      "@      &@      &@      @       @      @      "@      &@      (@      &@      ,@      .@      0@      2@      5@      5@      4@      4@      :@      9@      ;@      2@      &@       @       @        
�
Net/pred*�	    ����   ����?      Y@!  �T�A@)o|O�*�4@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��-Ա�L�����J�\��>	� �����T}�E��{��^?�l�P�`?*QH�x?o��5sz?#�+(�ŉ?�7c_XY�?�Rc�ݒ?^�S���?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      �?       @              �?       @      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?              �?      �?       @               @      �?       @      �?      @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�9@<e�H|k      ̃h	aRI��AB*�
�
	Net/h_out*�   ����?     @�@!  �8��Y@)���w0�A@2�        �-���q=�[^:��"?U�4@@�$?�T���C?a�$��{E?��bB�SY?�m9�H�[?E��{��^?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             H�@              �?              �?              �?      �?              �?              �?      �?               @      �?               @      �?      @      �?      @               @      �?       @      @      @      @      @      @       @       @      @      @      "@      @      $@      @      $@      "@      $@      @      @      $@      @      $@      (@      *@      (@      .@      2@      0@      4@      5@      4@      6@      9@      9@      >@      .@      &@       @       @        
�
Net/pred*�	    R��    ��?      Y@!  `��@@)Ի�33@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����7c_XY��#�+(�ŉ����J�\������=����lDZrS�nK���LQ�uWy��r?hyO�s?#�+(�ŉ?�7c_XY�?���&�?�Rc�ݒ?�v��ab�?�/��?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?       @       @               @      �?      �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?      �?      �?      �?      �?      �?       @      �?      �?       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1�+?<H��      ί�!	�lRI��AC*�
�
	Net/h_out*�    x��?     @�@! ��q�Y@)���M�A@2�        �-���q=���#@?�!�A?<DKc��T?ܗ�SsW?�l�P�`?���%��b?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             P�@              �?              �?               @              �?      �?              �?               @      �?               @      �?       @      �?              @      �?       @       @      @       @      @      @      @      @      @      @      @      "@      @      @      $@       @      &@      $@      @      "@       @       @      "@      (@      *@      ,@      ,@      2@      0@      4@      6@      3@      6@      8@      :@      <@      0@      &@       @       @        
�
Net/pred*�	   @����    ���?      Y@!  ���A@)�m��d�4@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���&b՞
�u�hyO�s��m9�H�[?E��{��^?���T}?>	� �?#�+(�ŉ?�7c_XY�?^�S���?�"�uԖ?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      �?       @      �?               @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @               @               @               @       @      �?       @       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�,><��5��      �ɟV	�wRI��AD*�
�
	Net/h_out*�    4��?     @�@! �5!�vY@)�9�/ЂA@2�        �-���q=+A�F�&?I�I�)�(?���#@?�!�A?�m9�H�[?E��{��^?�l�P�`?���%��b?P}���h?ߤ�(g%k?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             X�@              �?              �?              �?              �?              �?              �?              �?       @      �?      �?      �?       @      �?       @       @       @      @      �?       @      @      @      @       @      @      @       @      @      @      @      &@      @      &@       @      &@      @      @      "@      $@      $@      &@      *@      &@      0@      2@      1@      3@      5@      6@      5@      8@      ;@      ;@      1@      "@       @       @        
�
Net/pred*�	   ��n��    w�?      Y@!  X���@@)���&3@2����g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ��7c_XY��#�+(�ŉ�����=���>	� �����%��b��l�P�`�&b՞
�u?*QH�x?���J�\�?-Ա�L�?�Rc�ݒ?^�S���?��<�A��?�v��ab�?�/��?�uS��a�?�/�*>�?�g���w�?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      @      �?      �?       @      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?      �?               @               @               @      �?       @       @       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1?p=<�G      ���I	߁RI��AE*�
�
	Net/h_out*�   ���?     @�@! �%9mY@)�5�f�A@2�        �-���q=K+�E���>jqs&\��>��%>��:?d�\D�X=?�T���C?a�$��{E?�l�P�`?���%��b?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             P�@              �?              �?              �?              @              �?               @               @               @              @              �?      @      @       @      �?       @      @      @      @      @      @      @      @      @      @      @      "@      @      &@       @      (@      @      @       @      $@      &@      $@      (@      *@      .@      2@      1@      4@      5@      4@      4@      :@      :@      :@      0@      &@       @       @        
�
Net/pred*�	    �?��   ����?      Y@!  �O��A@)G�b�4@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�-Ա�L�����J�\�����T}�o��5sz����%��b��l�P�`��l�P�`?���%��b?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�"�uԖ?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      �?       @      �?              �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?               @               @               @               @       @      �?       @       @       @       @       @      @      @       @      @      @      @      @      @      @      @       @      @        

loss_1߁<<Ha�z{      �zny	M�RI��AF*�
�
	Net/h_out*�    ���?     @�@! @ ��IY@)�&�4�YA@2�        �-���q=��bȬ�0?��82?�u�w74?�m9�H�[?E��{��^?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             `�@              �?      �?              �?              �?              �?      @      �?               @              �?              @       @      �?       @       @      �?       @      @      @      @      @      @       @      @      @      @      @      @      @       @      $@      $@      $@      @      @      "@      @      &@      *@      &@      (@      1@      0@      2@      3@      4@      5@      6@      7@      ;@      :@      2@      "@       @       @        
�
Net/pred*�	    �n��   �B!�?      Y@!  P��@@)�J�*3@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�>	� �����T}����%��b��l�P�`�o��5sz?���T}?���J�\�?-Ա�L�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?       @      �?      �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?               @               @               @       @      �?       @       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @      �?        

loss_1O�;<�U�[      �t\�	��RI��AG*�
�
	Net/h_out*�   @c��?     @�@! �6kM@Y@)�V�%:ZA@2�        �-���q=�T���C?a�$��{E?E��{��^?�l�P�`?���%��b?�N�W�m?;8�clp?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             h�@              �?              �?      �?               @              �?      �?       @      �?      �?      �?      �?       @               @      @      �?      �?       @      @       @      @       @      @      @      @      @      @      @      @      @      $@       @      &@      "@       @      @       @      "@      &@      &@      *@      *@      ,@      1@      3@      3@      5@      3@      6@      8@      :@      :@      1@      $@       @       @        
�
Net/pred*�	   �oH��    1��?      Y@!  ��4�A@)�/C��4@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L������=���>	� ��&b՞
�u�hyO�s��7Kaa+�I�I�)�(��l�P�`?���%��b?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?}Y�4j�?��<�A��?�uS��a�?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?       @      �?      �?              �?      �?      �?              �?               @              �?              �?              �?              �?              �?               @              �?      �?              �?      �?      �?               @               @               @      �?      �?       @      �?       @       @       @       @       @      @      @       @      @      @      @      @      @      @      @       @      @        

loss_1��:<�����      ��	��RI��AH*�
�	
	Net/h_out*�    Y��?     @�@!  �K�&Y@)�����6A@2�        �-���q=�vV�R9?��ڋ?�u�w74?��%�V6?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             `�@              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?              �?      �?      �?               @       @       @      �?      @       @      @       @      @      @      @      @      @      @      @      @      @      @      "@       @      "@      $@      "@      @      "@      "@      @      &@      (@      *@      &@      0@      1@      2@      3@      4@      6@      5@      8@      :@      :@      1@      $@      @       @        
�
Net/pred*�	   @�ܧ�   @�/�?      Y@!  P��@@)��ư3@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY�����T}�o��5sz�ߤ�(g%k�P}���h����T}?>	� �?����=��?���J�\�?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?      �?       @      �?      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @               @      �?      �?               @      �?      �?       @      �?       @       @       @       @       @      @      @       @      @      @      @      @      @      @      @      @      �?        

loss_1A�9<���ʫ      ί�!	ШRI��AI*�
�
	Net/h_out*�   ཿ�?     @�@! �C��Y@)8P�9A@2�        �-���q=��%>��:?d�\D�X=?
����G?�qU���I?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             `�@              �?              �?               @              �?              �?               @              �?      �?       @              �?              �?       @      �?      �?      @       @       @      @      @      �?      @      @      @      @      @      @       @      @      @      "@      @      $@      $@      $@       @      "@      @       @      &@      &@      ,@      (@      .@      1@      2@      2@      5@      5@      5@      8@      :@      :@      0@      $@       @       @        
�
Net/pred*�	    �E��   ���?      Y@!  �,��A@)��%�4@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���o��5sz�*QH�x�&b՞
�u�Tw��Nof�5Ucv0ed��l�P�`?���%��b?5Ucv0ed?Tw��Nof?#�+(�ŉ?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?`��a�8�?�/�*>�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @              �?       @              �?               @              �?              �?              �?      �?              �?              �?              �?              �?      �?               @               @              �?      �?      �?               @               @               @      �?      �?       @      �?       @       @       @       @       @      @      @       @      @      @      @      @      @      @      @       @      @        

loss_1�9<�VD>�      ���I	#�RI��AJ*�
�
	Net/h_out*�   ����?     @�@!  vؐ�X@)��&9�A@2�        �-���q=��%>��:?d�\D�X=?a�$��{E?
����G?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?E��{��^?�l�P�`?ߤ�(g%k?�N�W�m?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             x�@              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @       @      @              @       @      @      �?       @      �?      @      @      @      @      @      @      @      @      @      "@       @      $@       @      &@      @      "@       @       @      $@      (@      &@      ,@      .@      0@      3@      2@      6@      4@      5@      9@      9@      :@      0@      $@      @       @        
�
Net/pred*�	   �&Ԧ�   �Z4�?      Y@!  Xp��@@)y	x��$3@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�o��5sz�*QH�x�5Ucv0ed����%��b�>	� �?����=��?���J�\�?^�S���?�"�uԖ?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?       @      �?       @      �?      �?       @              �?      �?              �?      �?              �?              �?              �?      �?               @              �?      �?              �?              �?      �?      �?      �?      �?      �?               @       @               @      �?       @       @       @       @       @      @      @       @      @      @      @      @      @      @      @      @      �?        

loss_1=8<L����      ��͋	ӼRI��AK*�
�	
	Net/h_out*�   ����?     @�@! �٦�X@)A�LlA@2�        �-���q=uܬ�@8?��%>��:?d�\D�X=?���#@?�qU���I?IcD���L?k�1^�sO?��bB�SY?�m9�H�[?���%��b?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             p�@              �?              �?              �?      �?              �?              �?              �?               @              �?              �?       @      �?      �?      �?       @      @      @       @       @      �?      �?      @      @      @      @      @      @      @      @      @      "@      @      &@      $@      &@      @      @       @      "@      "@      (@      (@      *@      0@      .@      3@      3@      5@      5@      3@      9@      ;@      8@      0@      $@       @       @        
�
Net/pred*�	    ����   �U�?      Y@!  �;9�A@)2�>���4@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\�����T}�o��5sz�uWy��r�;8�clp��N�W�m�k�1^�sO�IcD���L����%��b?5Ucv0ed?�N�W�m?;8�clp?#�+(�ŉ?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?      �?               @               @      �?               @      �?      �?       @               @       @      �?       @       @       @      @      @      @       @      @      @      @      @      @      @      @       @      @        

loss_1�h7<6�P�+      ͗i�	��RI��AL*�
�
	Net/h_out*�   �z��?     @�@!  ����X@)�R���@@2�        �-���q=���#@?�!�A?�l�P�`?���%��b?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?               @               @      �?      �?      �?      @       @       @      �?       @      �?      @      �?      @      @       @      @      @      @      @      @       @      @      @      $@      "@      $@      $@      @      "@      @      "@      (@      "@      *@      (@      0@      2@      .@      5@      4@      4@      6@      9@      9@      9@      1@      "@      @       @        
�
Net/pred*�	   @�G��   @KD�?      Y@!  �3Ļ@@)n��}x3@2��g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�&b՞
�u�hyO�s��N�W�m�ߤ�(g%k�>	� �?����=��?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?       @      �?       @      �?      �?               @      �?              �?               @              �?              �?               @              �?              �?               @              �?      �?              �?      �?      �?      �?      �?       @               @               @       @      �?       @       @       @      @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_1��6<s���      �̔�	��RI��AM*�
�
	Net/h_out*�   `J��?     @�@!  #��X@)���h;�@@2�        �-���q=��82?�u�w74?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?              �?              �?              �?      �?               @      �?               @      �?      �?      @      @      @               @      @       @      @      @      @      @      @      @      @      @      @      @      $@      "@      &@      $@      @       @      @      $@      $@      &@      (@      *@      0@      .@      2@      5@      4@      4@      4@      9@      ;@      7@      1@      $@      @       @        
�
Net/pred*�	    �Ԛ�   @-�?      Y@!  �q��A@)��۽4@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� ��&b՞
�u�hyO�s�Tw��Nof�5Ucv0ed����%��b��l�P�`��qU���I?IcD���L?�N�W�m?;8�clp?&b՞
�u?*QH�x?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?               @               @               @      �?      �?      �?       @      �?      �?       @      �?       @       @       @      @      @      @       @      @      @      @      @      @      @      @       @      @        

loss_1��5<]�T�[      �t\�	��RI��AN*�
�
	Net/h_out*�    ���?     @�@!  ��)�X@)h���@@2�        �-���q=�T���C?a�$��{E?�qU���I?IcD���L?�l�P�`?���%��b?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?      �?      �?              �?      �?      �?      �?      �?       @       @      �?       @      �?      @      �?       @      @      @      @      �?      @      @      @      @      @      @      @      "@      @      $@      $@      (@      @       @      $@      @      (@      &@      (@      *@      ,@      2@      2@      2@      4@      4@      6@      9@      9@      9@      0@      "@      @       @        
�
Net/pred*�	   @h9��    �K�?      Y@!  �a�@@)nt�('3@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��-Ա�L���N�W�m�ߤ�(g%k�P}���h�����=��?���J�\�?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @      �?      �?       @      �?      �?               @      �?              �?      �?      �?              �?      �?               @              �?      �?              �?      �?              �?      �?              �?      �?      �?      �?      �?       @               @      �?      �?       @      �?       @       @       @      @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_1�#5<#�*�      ����	��RI��AO*�
�
	Net/h_out*�   ����?     @�@! ���;�X@)��_u��@@2�        �-���q=
����G?�qU���I?nK���LQ?�lDZrS?���%��b?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?      �?              �?               @      �?              �?              �?      �?       @               @      @       @               @      @      @       @      @      @      @      @      @      @      @      @      @      @      "@      $@       @      ,@      @      @      "@      "@      $@      (@      *@      &@      ,@      3@      .@      4@      5@      4@      6@      8@      :@      7@      0@      $@      @       @        
�
Net/pred*�	    �j��   �2�?      Y@!  ���A@)7#R���4@2���<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�-Ա�L�����J�\�����T}�o��5sz�;8�clp��N�W�m�ܗ�SsW�<DKc��T�a�$��{E��T���C����%��b?5Ucv0ed?&b՞
�u?*QH�x?o��5sz?���T}?�7c_XY�?�#�h/�?}Y�4j�?��<�A��?�v��ab�?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?               @              �?      �?              �?      �?              �?      �?               @               @      �?      �?      �?       @      �?      �?       @      �?       @       @       @      @      @      @      @       @      @      @      @      @      @      @      @      @        

loss_12`4<�,��      �ɟV	�RI��AP*�
�
	Net/h_out*�   `���?     @�@! @E#��X@)+C����@@2�        �-���q=
����G?�qU���I?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?              �?      �?              �?              �?       @               @      @               @       @      �?       @      @       @      @      @      @       @      @      @      @      @      @      @      "@      @      $@      $@      $@       @      @      @      "@      (@      &@      &@      *@      .@      1@      4@      1@      4@      4@      6@      9@      9@      9@      .@      "@      @       @        
�
Net/pred*�	   �~���   �EW�?      Y@!  @=��@@)Tɲ��3@2��/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��;8�clp��N�W�m�Tw��Nof�5Ucv0ed�>	� �?����=��?���J�\�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?       @      �?       @      �?              �?       @              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?               @               @              �?      �?      �?       @               @      �?      �?       @      �?       @       @       @      @      @      @      @       @      @      @      @      @      @      @       @      �?        

loss_1��3<^8a@�      �̔�	{�RI��AQ*�
�
	Net/h_out*�   @��?     @�@! @����X@)?㘪�@@2�        �-���q=uܬ�@8?��%>��:?<DKc��T?ܗ�SsW?5Ucv0ed?Tw��Nof?P}���h?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?      �?               @               @              �?              @      �?       @      @       @      �?      �?      @       @      @      @      @       @      @      @      @      @      @      @      "@      @      &@      @      *@      @      @       @      $@      $@      (@      &@      *@      ,@      2@      2@      2@      4@      6@      4@      9@      ;@      6@      .@      $@      @       @        
�	
Net/pred*�		    ����   `#G�?      Y@!  0�Q�A@)g��4@2�}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L������=���>	� ��&b՞
�u�hyO�s�E��{��^��m9�H�[���%�V6?uܬ�@8?�lDZrS?<DKc��T?;8�clp?uWy��r?���T}?>	� �?����=��?���J�\�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?               @      �?      �?      �?       @      �?      �?       @      �?       @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�3<���      ����	nRI��AR*�
�
	Net/h_out*�   @}��?     @�@! ���FvX@)��<g�@@2�        �-���q=�qU���I?IcD���L?k�1^�sO?�m9�H�[?E��{��^?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?      �?              �?              �?              �?              �?      �?              �?       @      @       @              �?      �?       @      �?       @      @      �?      @      @      @      @      @      @      @      @       @      @       @      "@      "@      "@      &@       @       @      @      &@       @      &@      *@      ,@      .@      .@      3@      3@      4@      3@      7@      8@      9@      9@      0@       @      @       @        
�
Net/pred*�	   �[��   @�`�?      Y@!   ��@@)�Ww�3@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���hyO�s�uWy��r�E��{��^��m9�H�[����T}?>	� �?����=��?���J�\�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @       @      �?      �?      �?      �?      �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @      �?      �?      �?       @      �?       @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @       @      �?        

loss_17�2< ��+	      �g��	�RI��AS*�
�
	Net/h_out*�   `$��?     @�@! @n*�pX@)�v��l�@@2�        �-���q=���#@?�!�A?
����G?�qU���I?ܗ�SsW?��bB�SY?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?              �?              �?      �?              �?      �?              @       @      �?      @              �?       @       @       @      @      @      @       @      @      @      @      @      @       @      @       @       @      $@      "@      &@      "@      @      @      "@      $@      (@      &@      *@      1@      .@      2@      4@      4@      4@      4@      9@      ;@      6@      0@      "@      @       @        
�	
Net/pred*�		    �핿    W�?      Y@!  0&<�A@)��!��4@2��"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=������T}�o��5sz�*QH�x�ߤ�(g%k�P}���h���ڋ��vV�R9��m9�H�[?E��{��^?Tw��Nof?P}���h?*QH�x?o��5sz?����=��?���J�\�?-Ա�L�?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?              �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?               @               @       @      �?      �?       @      �?      �?       @       @       @      @      @      @      @       @      @      @      @      @      @      @      @      @        

loss_1�1<c��_�      ���I	>RI��AT*�
�
	Net/h_out*�   ����?     @�@! �I�GPX@)֝��m@@2�        �-���q=nK���LQ?�lDZrS?5Ucv0ed?Tw��Nof?P}���h?�N�W�m?;8�clp?uWy��r?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?       @              �?      �?              �?       @       @               @               @      �?      �?      @      @       @       @      @      �?      @      @      @      @      @      @      @       @       @      @      &@      "@      (@      @      "@      $@       @       @      (@      (@      ,@      .@      .@      2@      3@      3@      4@      7@      7@      :@      8@      0@       @      @       @        
�
Net/pred*�	    ��   �f�?      Y@!  @K�@@)2S���%3@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�-Ա�L�����J�\������=������T}�o��5sz��N�W�m�ߤ�(g%k���%�V6��u�w74�>	� �?����=��?���J�\�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @      �?      �?       @              �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?               @               @               @      �?      �?      �?      �?       @      �?       @      �?       @      @      @       @      @      @      @      @      @      @      @      @      @       @      �?        

loss_11<t-�	      �X�t	� RI��AU*�
�
	Net/h_out*�    ���?     @�@! @�5�IX@)��#h�q@@2�        �-���q=��[�?1��a˲?IcD���L?k�1^�sO?��bB�SY?�m9�H�[?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?      �?              �?               @               @      �?               @               @       @               @      @       @      @       @      @      @       @      @      @      @      @      @      @      @      $@      @      $@      "@      &@       @      @      @      $@      "@      &@      ,@      (@      0@      .@      2@      4@      3@      4@      6@      7@      ;@      7@      .@      "@      @       @        
�	
Net/pred*�		    ���    �Y�?      Y@!  0l��A@)3���
�4@2�^�S�����Rc�ݒ����&���#�h/���7c_XY��eiS�m��-Ա�L������=���>	� ��o��5sz�*QH�x�&b՞
�u�hyO�s�E��{��^��m9�H�[��lDZrS?<DKc��T?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?���T}?>	� �?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?�Rc�ݒ?^�S���?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @      �?      �?       @      �?       @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      @      @        

loss_1L0<Ts���      ��͋	�*RI��AV*�
�
	Net/h_out*�   `���?     @�@! ���=4X@)l'_{1R@@2�        �-���q=�qU���I?IcD���L?k�1^�sO?<DKc��T?ܗ�SsW?Tw��Nof?P}���h?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?      �?              �?              �?              �?      �?              �?      �?       @      �?      �?              @      @      �?       @      �?       @      @      @      �?      @      @      @      @      @      @      @       @      @       @      &@       @      (@      @       @      $@      $@      "@      &@      $@      .@      ,@      0@      4@      1@      3@      6@      4@      9@      :@      7@      0@      @      @       @        
�	
Net/pred*�		    Z���   ��s�?      Y@!  0p�@@)~��	$3@2�`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� �����T}�*QH�x�&b՞
�u�5Ucv0ed����%��b�a�$��{E?
����G?���T}?>	� �?����=��?���J�\�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @      �?      �?               @      �?              �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @               @      �?      �?      �?       @      �?       @      �?      �?       @      @      @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1=�/<)�Cl+	      �g��	85RI��AW*�
�
	Net/h_out*�   ����?     @�@! �G�/X@))��|+X@@2�        �-���q=��bȬ�0?��82?E��{��^?�l�P�`?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?      �?              �?               @              �?              �?       @       @               @      @       @       @       @      �?      @      @      @      @      @      @      @      @      @      @       @      @      (@       @      *@      @       @       @       @      "@      (@      *@      (@      0@      0@      1@      3@      3@      6@      5@      7@      :@      8@      ,@      "@      @       @        
�	
Net/pred*�		    t��   ��g�?      Y@!  m�A@)����4@2�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��o��5sz�*QH�x�hyO�s�uWy��r��N�W�m�ߤ�(g%k���bȬ�0���VlQ.�Tw��Nof?P}���h?�N�W�m?;8�clp?*QH�x?o��5sz?����=��?���J�\�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @      �?       @      �?       @      �?      �?       @      @      @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�!/<��R��      ��	�?RI��AX*�
�
	Net/h_out*�   ����?     @�@! ��%�X@)y��%8@@2�        �-���q=��bB�SY?�m9�H�[?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ȁ@              �?              �?      �?              �?               @              �?      @              �?      @               @      �?      �?       @      @      @      @      @       @      @      @      @      @      @       @      @      @      $@      $@      &@       @      @      "@      $@      "@      &@      (@      *@      .@      1@      1@      4@      4@      3@      5@      9@      9@      7@      1@      @      @       @        
�	
Net/pred*�		   �1�    �y�?      Y@!  �*�@@)4��5� 3@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�o��5sz�hyO�s�uWy��r�ܗ�SsW�<DKc��T�<DKc��T?ܗ�SsW?o��5sz?���T}?���J�\�?-Ա�L�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @      �?      �?       @              �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @               @      �?      �?      �?       @      �?       @       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�.<Є��+	      �g��	KRI��AY*�
�
	Net/h_out*�   ���?     @�@! �Yw�X@)r7��>@@2�        �-���q=��ڋ?�.�?��%>��:?d�\D�X=?���#@?�!�A?�l�P�`?���%��b?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?      �?              �?              �?      �?               @              �?               @      �?      @      �?       @      �?      @      �?      �?      @      @      @      @      @       @      @      @      @      @      "@      @      @      &@       @      (@      @       @      "@       @      &@      &@      &@      *@      0@      .@      2@      4@      2@      6@      4@      8@      :@      7@      ,@      "@      @       @        
�	
Net/pred*�		    �`��   ��s�?      Y@!  @��A@)+�^ދ�4@2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� ��hyO�s�uWy��r��N�W�m�ߤ�(g%k�E��{��^��m9�H�[�<DKc��T?ܗ�SsW?;8�clp?uWy��r?hyO�s?&b՞
�u?���T}?>	� �?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?      �?      �?       @      �?       @      �?       @       @      �?       @       @      @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1�	.<)QJ�      ����	�VRI��AZ*�
�
	Net/h_out*�    ��?     @�@! @����W@)�>�@@2�        �-���q=+A�F�&?I�I�)�(?�m9�H�[?E��{��^?ߤ�(g%k?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?               @              �?      �?              @               @      �?      @      �?      �?       @              �?      �?       @      @      @      @       @      @      @      @      @      @      @      @      @       @      "@      @      &@      $@       @       @      "@       @      &@      $@      &@      0@      *@      1@      2@      3@      3@      3@      6@      9@      8@      8@      0@      @      @       @        
�	
Net/pred*�		   �    ����?      Y@!  ��I�@@)o��/3@2��uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m������=���>	� ��*QH�x�&b՞
�u�5Ucv0ed����%��b�
����G?�qU���I?Tw��Nof?P}���h?���T}?>	� �?-Ա�L�?eiS�m�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @      �?               @      �?              �?               @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?               @      �?      �?      �?       @      �?       @       @       @       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1eW-<ѿ>C�      ��͋	�`RI��A[*�
�
	Net/h_out*�   �!��?     @�@!  ���W@)�e�F�@@2�        �-���q=
����G?�qU���I?�m9�H�[?E��{��^?���%��b?5Ucv0ed?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ȁ@              �?              �?              �?               @              �?              �?              �?      �?       @       @       @       @      �?       @              �?      @       @      @       @      @      �?      @      @      @      @       @      @      @      $@      @      "@      *@       @       @       @       @      &@      &@      (@      *@      .@      0@      2@      3@      3@      5@      4@      8@      :@      6@      0@      @      @       @        
�	
Net/pred*�	    �M��   `zt�?      Y@!  p.��A@):�*�4@2����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\�����T}�o��5sz�;8�clp��N�W�m�P}���h�Tw��Nof�a�$��{E��T���C����%��b?5Ucv0ed?&b՞
�u?*QH�x?����=��?���J�\�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?               @              �?              �?      �?              �?              �?              �?      �?               @               @              �?      �?      �?       @       @      �?      �?       @       @       @       @       @       @       @      @      @      @      @      @      @      @      @      @      @      @        

loss_1a�,<?f�$+	      �g��	ojRI��A\*�
�	
	Net/h_out*�   �-��?     @�@! �v�"�W@)�4�Z @@2�        �-���q=ji6�9�?�S�F !?�7Kaa+?��VlQ.?��%>��:?d�\D�X=?E��{��^?�l�P�`?���%��b?5Ucv0ed?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?              �?              �?              �?      �?      �?              @       @      �?              �?               @       @      �?      @      @      @       @       @      @      @      @      @      @      @      @      @       @      $@      "@      $@      $@      "@      @      @      &@      $@      $@      *@      *@      *@      1@      2@      3@      3@      3@      6@      9@      9@      9@      ,@      @      @       @        
�	
Net/pred*�	    �j��   @v��?      Y@!  ��6�@@)kU6!]-3@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�hyO�s�uWy��r�ܗ�SsW�<DKc��T��m9�H�[?E��{��^?�N�W�m?;8�clp?*QH�x?o��5sz?eiS�m�?#�+(�ŉ?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      �?      �?      �?      �?      �?      �?               @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?               @      �?      �?       @      �?      �?       @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1�S,<�]��      �⦜	}sRI��A]*�
�
	Net/h_out*�    ���?     @�@! �R��W@) ��@@2�        �-���q=�h���`�>�ߊ4F��>��VlQ.?��bȬ�0?k�1^�sO?nK���LQ?�lDZrS?5Ucv0ed?Tw��Nof?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?      �?              �?               @      �?              �?              �?       @      �?      @              �?      �?      �?       @       @       @      @      @      @       @      @      @      @      @      @      @      @      "@      @      &@      "@      &@      "@      @      "@      "@      $@      &@      &@      *@      0@      0@      2@      3@      3@      4@      5@      9@      8@      6@      0@      @      @       @        
�
Net/pred*�	    ࢍ�   �`��?      Y@!  �>�A@)��;�4@2��#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L������=���>	� ��&b՞
�u�hyO�s����%��b��l�P�`�E��{��^��qU���I?IcD���L?�N�W�m?;8�clp?o��5sz?���T}?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?      �?              �?              �?              �?      �?              �?              �?               @              �?              �?      �?      �?              �?              �?      �?              �?              �?              �?      �?               @               @              �?      �?      �?       @       @      �?      �?       @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1��+<j��U	      �X�t	�~RI��A^*�
�
	Net/h_out*�   ����?     @�@! �!ń�W@)�/�m��?@2�        �-���q=a�$��{E?
����G?���%��b?5Ucv0ed?Tw��Nof?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ؁@              �?              �?      �?              �?      �?      �?              �?       @      �?      �?              �?       @      �?      �?      �?      @      �?      @      @       @      @      @      @      @      @      @      @      @       @      @       @       @      *@      "@      "@      @      "@       @      $@      (@      &@      ,@      0@      ,@      3@      1@      3@      5@      5@      9@      8@      :@      *@      @      @       @        
�	
Net/pred*�		    ����    ���?      Y@!  ���@@)��9S�:3@2��/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���*QH�x�&b՞
�u��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed��!�A?�T���C?�N�W�m?;8�clp?hyO�s?&b՞
�u?o��5sz?���T}?#�+(�ŉ?�7c_XY�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?       @              �?       @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?      �?       @               @      �?      �?       @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1��*<�Kx�      ��	�RI��A_*�
�
	Net/h_out*�   �5��?     @�@!  Z̈́�W@)�p�.�?@2�        �-���q=IcD���L?k�1^�sO?�lDZrS?<DKc��T?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ؁@              �?              �?              �?              �?              �?      �?               @      �?       @       @      �?      �?      �?      �?      �?      @       @       @       @      @       @       @      @      @      @      @      @      @      @      @      "@      $@       @      (@      $@       @      @      &@       @      &@      *@      (@      0@      0@      2@      3@      2@      5@      4@      :@      7@      8@      .@      @      @       @        
�
Net/pred*�	    ����   ���?      Y@!  �A��A@)t�!f�4@2��7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\��>	� �����T}�uWy��r�;8�clp�ܗ�SsW�<DKc��T��lDZrS��m9�H�[?E��{��^?hyO�s?&b՞
�u?���T}?>	� �?����=��?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?              �?      �?              �?              �?              �?      �?              �?              �?              �?      �?              �?              �?      �?      �?              �?       @              �?      �?      �?              �?               @               @               @      �?      �?      �?       @      �?      �?       @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1�;*<�%y	      �X�t		�RI��A`*�
�
	Net/h_out*�    ʠ�?     @�@!  ����W@)s:����?@2�        �-���q=k�1^�sO?nK���LQ?5Ucv0ed?Tw��Nof?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?               @              �?              �?      �?               @      �?      �?      �?              �?               @      @              �?      @      @      @      @       @      @      @      @      @      @      @      @      @       @       @      "@      (@       @      "@       @      $@       @      $@      $@      (@      .@      ,@      .@      2@      3@      2@      4@      7@      7@      :@      8@      *@      @      @       @        
�	
Net/pred*�		    ����   �u��?      Y@!  ����@@)��*j7:3@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L��>	� �����T}�uWy��r�;8�clp�5Ucv0ed����%��b�<DKc��T��lDZrS�E��{��^?�l�P�`?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?�"�uԖ?}Y�4j�?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�               @      �?      �?               @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?      �?      �?       @               @      �?      �?       @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1�)<��,��      ί�!	��RI��Aa*�
�
	Net/h_out*�   ����?     @�@! �nX��W@)�d��>�?@2�        �-���q=�S�F !?�[^:��"?ܗ�SsW?��bB�SY?P}���h?ߤ�(g%k?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?       @      �?              @               @      �?      �?      �?      @       @      �?      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      "@      $@      "@      &@      "@      "@       @      "@      "@      $@      (@      .@      .@      *@      4@      3@      2@      5@      4@      9@      9@      7@      ,@      @      @       @        
�
Net/pred*�	    ���   @ɋ�?      Y@!   �-�A@);Lm�V�4@2�#�+(�ŉ�eiS�m�����J�\������=���*QH�x�&b՞
�u�P}���h�Tw��Nof���%�V6��u�w74��u�w74?��%�V6?ߤ�(g%k?�N�W�m?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @              �?      �?      �?              �?      �?      �?               @               @      �?      �?      �?       @       @      �?      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1��(<y_��+	      �g��	K�RI��Ab*�
�
	Net/h_out*�   � ��?     @�@!  ��d�W@)��Ν�S?@2�        �-���q=a�$��{E?
����G?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?Tw��Nof?P}���h?ߤ�(g%k?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?              �?              �?       @              �?              �?      �?               @       @      �?               @      �?      �?      @       @      @       @      @       @      @      @      @      @      @       @      @      @       @       @      $@      "@      &@      @       @      $@      $@      $@      $@      *@      *@      (@      2@      2@      4@      0@      3@      9@      6@      9@      8@      *@      @      @       @        
�	
Net/pred*�		    |˛�   @���?      Y@!  ��@@)��f�;3@2��v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m�����J�\������=���*QH�x�&b՞
�u�Tw��Nof�5Ucv0ed�k�1^�sO�IcD���L����#@?�!�A?ߤ�(g%k?�N�W�m?o��5sz?���T}?>	� �?���J�\�?-Ա�L�?�#�h/�?���&�?�Rc�ݒ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?      �?      �?      �?               @              �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?      �?              �?      �?               @              �?      �?      �?       @               @       @               @       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1wY(<��3M�      ���I	`�RI��Ac*�
�
	Net/h_out*�   ����?     @�@!  �0�zW@)T��KU?@2�        �-���q=�7Kaa+?��VlQ.?�m9�H�[?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�             ��@              �?              �?      �?               @      �?              �?      �?      �?              �?      �?      �?       @       @      �?      �?       @      �?       @      @      @       @      @      @      @      @      @      @      @      @      @      "@      @      (@      "@      (@      @      "@      $@      @      "@      &@      *@      0@      (@      0@      3@      1@      3@      4@      6@      8@      9@      6@      ,@      @      @       @        
�
Net/pred*�	    ���   ����?      Y@!  p��A@)��܊a�4@2�-Ա�L�����J�\��>	� �����T}�uWy��r�;8�clp��lDZrS�nK���LQ�IcD���L?k�1^�sO?E��{��^?�l�P�`?hyO�s?&b՞
�u?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�/��?�uS��a�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?               @              �?      �?       @               @              �?      �?      �?      �?               @               @      �?      �?      �?       @       @      �?      �?       @       @       @       @       @      @       @      @      @      @      @      @      @      @      @      @      @        

loss_1�'<��Z�