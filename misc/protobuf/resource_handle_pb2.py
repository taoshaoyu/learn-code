# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: resource_handle.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import tensor_shape_pb2 as tensor__shape__pb2
import types_pb2 as types__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='resource_handle.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n\x15resource_handle.proto\x12\ntensorflow\x1a\x12tensor_shape.proto\x1a\x0btypes.proto\"\xa5\x02\n\x13ResourceHandleProto\x12\x0e\n\x06\x64\x65vice\x18\x01 \x01(\t\x12\x11\n\tcontainer\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x11\n\thash_code\x18\x04 \x01(\x04\x12\x17\n\x0fmaybe_type_name\x18\x05 \x01(\t\x12H\n\x11\x64types_and_shapes\x18\x06 \x03(\x0b\x32-.tensorflow.ResourceHandleProto.DtypeAndShape\x1a\x61\n\rDtypeAndShape\x12#\n\x05\x64type\x18\x01 \x01(\x0e\x32\x14.tensorflow.DataType\x12+\n\x05shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProtoJ\x04\x08\x07\x10\x08\x42\x87\x01\n\x18org.tensorflow.frameworkB\x0eResourceHandleP\x01ZVgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/resource_handle_go_proto\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensor__shape__pb2.DESCRIPTOR,types__pb2.DESCRIPTOR,])




_RESOURCEHANDLEPROTO_DTYPEANDSHAPE = _descriptor.Descriptor(
  name='DtypeAndShape',
  full_name='tensorflow.ResourceHandleProto.DtypeAndShape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tensorflow.ResourceHandleProto.DtypeAndShape.dtype', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.ResourceHandleProto.DtypeAndShape.shape', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=358,
)

_RESOURCEHANDLEPROTO = _descriptor.Descriptor(
  name='ResourceHandleProto',
  full_name='tensorflow.ResourceHandleProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device', full_name='tensorflow.ResourceHandleProto.device', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='container', full_name='tensorflow.ResourceHandleProto.container', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorflow.ResourceHandleProto.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hash_code', full_name='tensorflow.ResourceHandleProto.hash_code', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maybe_type_name', full_name='tensorflow.ResourceHandleProto.maybe_type_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtypes_and_shapes', full_name='tensorflow.ResourceHandleProto.dtypes_and_shapes', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_RESOURCEHANDLEPROTO_DTYPEANDSHAPE, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=364,
)

_RESOURCEHANDLEPROTO_DTYPEANDSHAPE.fields_by_name['dtype'].enum_type = types__pb2._DATATYPE
_RESOURCEHANDLEPROTO_DTYPEANDSHAPE.fields_by_name['shape'].message_type = tensor__shape__pb2._TENSORSHAPEPROTO
_RESOURCEHANDLEPROTO_DTYPEANDSHAPE.containing_type = _RESOURCEHANDLEPROTO
_RESOURCEHANDLEPROTO.fields_by_name['dtypes_and_shapes'].message_type = _RESOURCEHANDLEPROTO_DTYPEANDSHAPE
DESCRIPTOR.message_types_by_name['ResourceHandleProto'] = _RESOURCEHANDLEPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ResourceHandleProto = _reflection.GeneratedProtocolMessageType('ResourceHandleProto', (_message.Message,), dict(

  DtypeAndShape = _reflection.GeneratedProtocolMessageType('DtypeAndShape', (_message.Message,), dict(
    DESCRIPTOR = _RESOURCEHANDLEPROTO_DTYPEANDSHAPE,
    __module__ = 'resource_handle_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.ResourceHandleProto.DtypeAndShape)
    ))
  ,
  DESCRIPTOR = _RESOURCEHANDLEPROTO,
  __module__ = 'resource_handle_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ResourceHandleProto)
  ))
_sym_db.RegisterMessage(ResourceHandleProto)
_sym_db.RegisterMessage(ResourceHandleProto.DtypeAndShape)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\016ResourceHandleP\001ZVgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/resource_handle_go_proto\370\001\001'))
# @@protoc_insertion_point(module_scope)
