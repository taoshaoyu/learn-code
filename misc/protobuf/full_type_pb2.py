# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: full_type.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='full_type.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n\x0f\x66ull_type.proto\x12\ntensorflow\"\x7f\n\x0b\x46ullTypeDef\x12\'\n\x07type_id\x18\x01 \x01(\x0e\x32\x16.tensorflow.FullTypeId\x12%\n\x04\x61rgs\x18\x02 \x03(\x0b\x32\x17.tensorflow.FullTypeDef\x12\x0b\n\x01s\x18\x03 \x01(\tH\x00\x12\x0b\n\x01i\x18\x04 \x01(\x03H\x00\x42\x06\n\x04\x61ttr*\xc3\x04\n\nFullTypeId\x12\r\n\tTFT_UNSET\x10\x00\x12\x0b\n\x07TFT_VAR\x10\x01\x12\x0b\n\x07TFT_ANY\x10\x02\x12\x0f\n\x0bTFT_PRODUCT\x10\x03\x12\r\n\tTFT_NAMED\x10\x04\x12\x10\n\x0cTFT_FOR_EACH\x10\x14\x12\x10\n\x0cTFT_CALLABLE\x10\x64\x12\x0f\n\nTFT_TENSOR\x10\xe8\x07\x12\x0e\n\tTFT_ARRAY\x10\xe9\x07\x12\x11\n\x0cTFT_OPTIONAL\x10\xea\x07\x12\x10\n\x0bTFT_LITERAL\x10\xeb\x07\x12\x10\n\x0bTFT_ENCODED\x10\xec\x07\x12\r\n\x08TFT_BOOL\x10\xc8\x01\x12\x0e\n\tTFT_UINT8\x10\xc9\x01\x12\x0f\n\nTFT_UINT16\x10\xca\x01\x12\x0f\n\nTFT_UINT32\x10\xcb\x01\x12\x0f\n\nTFT_UINT64\x10\xcc\x01\x12\r\n\x08TFT_INT8\x10\xcd\x01\x12\x0e\n\tTFT_INT16\x10\xce\x01\x12\x0e\n\tTFT_INT32\x10\xcf\x01\x12\x0e\n\tTFT_INT64\x10\xd0\x01\x12\r\n\x08TFT_HALF\x10\xd1\x01\x12\x0e\n\tTFT_FLOAT\x10\xd2\x01\x12\x0f\n\nTFT_DOUBLE\x10\xd3\x01\x12\x11\n\x0cTFT_BFLOAT16\x10\xd7\x01\x12\x12\n\rTFT_COMPLEX64\x10\xd4\x01\x12\x13\n\x0eTFT_COMPLEX128\x10\xd5\x01\x12\x0f\n\nTFT_STRING\x10\xd6\x01\x12\x10\n\x0bTFT_DATASET\x10\xf6N\x12\x0f\n\nTFT_RAGGED\x10\xf7N\x12\x11\n\x0cTFT_ITERATOR\x10\xf8N\x12\x13\n\x0eTFT_MUTEX_LOCK\x10\xdaO\x12\x17\n\x12TFT_LEGACY_VARIANT\x10\xdbOB\x81\x01\n\x18org.tensorflow.frameworkB\x0e\x46ullTypeProtosP\x01ZPgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/full_type_go_proto\xf8\x01\x01\x62\x06proto3')
)

_FULLTYPEID = _descriptor.EnumDescriptor(
  name='FullTypeId',
  full_name='tensorflow.FullTypeId',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TFT_UNSET', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_VAR', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_ANY', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_PRODUCT', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_NAMED', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_FOR_EACH', index=5, number=20,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_CALLABLE', index=6, number=100,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_TENSOR', index=7, number=1000,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_ARRAY', index=8, number=1001,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_OPTIONAL', index=9, number=1002,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_LITERAL', index=10, number=1003,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_ENCODED', index=11, number=1004,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_BOOL', index=12, number=200,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_UINT8', index=13, number=201,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_UINT16', index=14, number=202,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_UINT32', index=15, number=203,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_UINT64', index=16, number=204,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_INT8', index=17, number=205,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_INT16', index=18, number=206,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_INT32', index=19, number=207,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_INT64', index=20, number=208,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_HALF', index=21, number=209,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_FLOAT', index=22, number=210,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_DOUBLE', index=23, number=211,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_BFLOAT16', index=24, number=215,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_COMPLEX64', index=25, number=212,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_COMPLEX128', index=26, number=213,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_STRING', index=27, number=214,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_DATASET', index=28, number=10102,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_RAGGED', index=29, number=10103,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_ITERATOR', index=30, number=10104,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_MUTEX_LOCK', index=31, number=10202,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TFT_LEGACY_VARIANT', index=32, number=10203,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=161,
  serialized_end=740,
)
_sym_db.RegisterEnumDescriptor(_FULLTYPEID)

FullTypeId = enum_type_wrapper.EnumTypeWrapper(_FULLTYPEID)
TFT_UNSET = 0
TFT_VAR = 1
TFT_ANY = 2
TFT_PRODUCT = 3
TFT_NAMED = 4
TFT_FOR_EACH = 20
TFT_CALLABLE = 100
TFT_TENSOR = 1000
TFT_ARRAY = 1001
TFT_OPTIONAL = 1002
TFT_LITERAL = 1003
TFT_ENCODED = 1004
TFT_BOOL = 200
TFT_UINT8 = 201
TFT_UINT16 = 202
TFT_UINT32 = 203
TFT_UINT64 = 204
TFT_INT8 = 205
TFT_INT16 = 206
TFT_INT32 = 207
TFT_INT64 = 208
TFT_HALF = 209
TFT_FLOAT = 210
TFT_DOUBLE = 211
TFT_BFLOAT16 = 215
TFT_COMPLEX64 = 212
TFT_COMPLEX128 = 213
TFT_STRING = 214
TFT_DATASET = 10102
TFT_RAGGED = 10103
TFT_ITERATOR = 10104
TFT_MUTEX_LOCK = 10202
TFT_LEGACY_VARIANT = 10203



_FULLTYPEDEF = _descriptor.Descriptor(
  name='FullTypeDef',
  full_name='tensorflow.FullTypeDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type_id', full_name='tensorflow.FullTypeDef.type_id', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='args', full_name='tensorflow.FullTypeDef.args', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='s', full_name='tensorflow.FullTypeDef.s', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='tensorflow.FullTypeDef.i', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
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
    _descriptor.OneofDescriptor(
      name='attr', full_name='tensorflow.FullTypeDef.attr',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=31,
  serialized_end=158,
)

_FULLTYPEDEF.fields_by_name['type_id'].enum_type = _FULLTYPEID
_FULLTYPEDEF.fields_by_name['args'].message_type = _FULLTYPEDEF
_FULLTYPEDEF.oneofs_by_name['attr'].fields.append(
  _FULLTYPEDEF.fields_by_name['s'])
_FULLTYPEDEF.fields_by_name['s'].containing_oneof = _FULLTYPEDEF.oneofs_by_name['attr']
_FULLTYPEDEF.oneofs_by_name['attr'].fields.append(
  _FULLTYPEDEF.fields_by_name['i'])
_FULLTYPEDEF.fields_by_name['i'].containing_oneof = _FULLTYPEDEF.oneofs_by_name['attr']
DESCRIPTOR.message_types_by_name['FullTypeDef'] = _FULLTYPEDEF
DESCRIPTOR.enum_types_by_name['FullTypeId'] = _FULLTYPEID
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FullTypeDef = _reflection.GeneratedProtocolMessageType('FullTypeDef', (_message.Message,), dict(
  DESCRIPTOR = _FULLTYPEDEF,
  __module__ = 'full_type_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.FullTypeDef)
  ))
_sym_db.RegisterMessage(FullTypeDef)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\016FullTypeProtosP\001ZPgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/full_type_go_proto\370\001\001'))
# @@protoc_insertion_point(module_scope)
