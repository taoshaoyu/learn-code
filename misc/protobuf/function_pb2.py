# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: function.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import attr_value_pb2 as attr__value__pb2
import node_def_pb2 as node__def__pb2
import op_def_pb2 as op__def__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='function.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n\x0e\x66unction.proto\x12\ntensorflow\x1a\x10\x61ttr_value.proto\x1a\x0enode_def.proto\x1a\x0cop_def.proto\"\xa8\x01\n\x12\x46unctionDefLibrary\x12)\n\x08\x66unction\x18\x01 \x03(\x0b\x32\x17.tensorflow.FunctionDef\x12)\n\x08gradient\x18\x02 \x03(\x0b\x32\x17.tensorflow.GradientDef\x12<\n\x14registered_gradients\x18\x03 \x03(\x0b\x32\x1e.tensorflow.RegisteredGradient\"\xc4\x06\n\x0b\x46unctionDef\x12$\n\tsignature\x18\x01 \x01(\x0b\x32\x11.tensorflow.OpDef\x12/\n\x04\x61ttr\x18\x05 \x03(\x0b\x32!.tensorflow.FunctionDef.AttrEntry\x12\x36\n\x08\x61rg_attr\x18\x07 \x03(\x0b\x32$.tensorflow.FunctionDef.ArgAttrEntry\x12P\n\x16resource_arg_unique_id\x18\x08 \x03(\x0b\x32\x30.tensorflow.FunctionDef.ResourceArgUniqueIdEntry\x12%\n\x08node_def\x18\x03 \x03(\x0b\x32\x13.tensorflow.NodeDef\x12-\n\x03ret\x18\x04 \x03(\x0b\x32 .tensorflow.FunctionDef.RetEntry\x12<\n\x0b\x63ontrol_ret\x18\x06 \x03(\x0b\x32\'.tensorflow.FunctionDef.ControlRetEntry\x1a\x42\n\tAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.tensorflow.AttrValue:\x02\x38\x01\x1a\x88\x01\n\x08\x41rgAttrs\x12\x38\n\x04\x61ttr\x18\x01 \x03(\x0b\x32*.tensorflow.FunctionDef.ArgAttrs.AttrEntry\x1a\x42\n\tAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.tensorflow.AttrValue:\x02\x38\x01\x1aP\n\x0c\x41rgAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12/\n\x05value\x18\x02 \x01(\x0b\x32 .tensorflow.FunctionDef.ArgAttrs:\x02\x38\x01\x1a:\n\x18ResourceArgUniqueIdEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12\r\n\x05value\x18\x02 \x01(\r:\x02\x38\x01\x1a*\n\x08RetEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x31\n\x0f\x43ontrolRetEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01J\x04\x08\x02\x10\x03\";\n\x0bGradientDef\x12\x15\n\rfunction_name\x18\x01 \x01(\t\x12\x15\n\rgradient_func\x18\x02 \x01(\t\"G\n\x12RegisteredGradient\x12\x15\n\rgradient_func\x18\x01 \x01(\t\x12\x1a\n\x12registered_op_type\x18\x02 \x01(\tB\x80\x01\n\x18org.tensorflow.frameworkB\x0e\x46unctionProtosP\x01ZOgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/function_go_proto\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[attr__value__pb2.DESCRIPTOR,node__def__pb2.DESCRIPTOR,op__def__pb2.DESCRIPTOR,])




_FUNCTIONDEFLIBRARY = _descriptor.Descriptor(
  name='FunctionDefLibrary',
  full_name='tensorflow.FunctionDefLibrary',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function', full_name='tensorflow.FunctionDefLibrary.function', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gradient', full_name='tensorflow.FunctionDefLibrary.gradient', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='registered_gradients', full_name='tensorflow.FunctionDefLibrary.registered_gradients', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=79,
  serialized_end=247,
)


_FUNCTIONDEF_ATTRENTRY = _descriptor.Descriptor(
  name='AttrEntry',
  full_name='tensorflow.FunctionDef.AttrEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.AttrEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.AttrEntry.value', index=1,
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
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=638,
  serialized_end=704,
)

_FUNCTIONDEF_ARGATTRS_ATTRENTRY = _descriptor.Descriptor(
  name='AttrEntry',
  full_name='tensorflow.FunctionDef.ArgAttrs.AttrEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.ArgAttrs.AttrEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.ArgAttrs.AttrEntry.value', index=1,
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
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=638,
  serialized_end=704,
)

_FUNCTIONDEF_ARGATTRS = _descriptor.Descriptor(
  name='ArgAttrs',
  full_name='tensorflow.FunctionDef.ArgAttrs',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='attr', full_name='tensorflow.FunctionDef.ArgAttrs.attr', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FUNCTIONDEF_ARGATTRS_ATTRENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=707,
  serialized_end=843,
)

_FUNCTIONDEF_ARGATTRENTRY = _descriptor.Descriptor(
  name='ArgAttrEntry',
  full_name='tensorflow.FunctionDef.ArgAttrEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.ArgAttrEntry.key', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.ArgAttrEntry.value', index=1,
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
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=845,
  serialized_end=925,
)

_FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY = _descriptor.Descriptor(
  name='ResourceArgUniqueIdEntry',
  full_name='tensorflow.FunctionDef.ResourceArgUniqueIdEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.ResourceArgUniqueIdEntry.key', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.ResourceArgUniqueIdEntry.value', index=1,
      number=2, type=13, cpp_type=3, label=1,
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
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=927,
  serialized_end=985,
)

_FUNCTIONDEF_RETENTRY = _descriptor.Descriptor(
  name='RetEntry',
  full_name='tensorflow.FunctionDef.RetEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.RetEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.RetEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=987,
  serialized_end=1029,
)

_FUNCTIONDEF_CONTROLRETENTRY = _descriptor.Descriptor(
  name='ControlRetEntry',
  full_name='tensorflow.FunctionDef.ControlRetEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.FunctionDef.ControlRetEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.FunctionDef.ControlRetEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1031,
  serialized_end=1080,
)

_FUNCTIONDEF = _descriptor.Descriptor(
  name='FunctionDef',
  full_name='tensorflow.FunctionDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='signature', full_name='tensorflow.FunctionDef.signature', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attr', full_name='tensorflow.FunctionDef.attr', index=1,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arg_attr', full_name='tensorflow.FunctionDef.arg_attr', index=2,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resource_arg_unique_id', full_name='tensorflow.FunctionDef.resource_arg_unique_id', index=3,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='node_def', full_name='tensorflow.FunctionDef.node_def', index=4,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ret', full_name='tensorflow.FunctionDef.ret', index=5,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='control_ret', full_name='tensorflow.FunctionDef.control_ret', index=6,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_FUNCTIONDEF_ATTRENTRY, _FUNCTIONDEF_ARGATTRS, _FUNCTIONDEF_ARGATTRENTRY, _FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY, _FUNCTIONDEF_RETENTRY, _FUNCTIONDEF_CONTROLRETENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=250,
  serialized_end=1086,
)


_GRADIENTDEF = _descriptor.Descriptor(
  name='GradientDef',
  full_name='tensorflow.GradientDef',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_name', full_name='tensorflow.GradientDef.function_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gradient_func', full_name='tensorflow.GradientDef.gradient_func', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=1088,
  serialized_end=1147,
)


_REGISTEREDGRADIENT = _descriptor.Descriptor(
  name='RegisteredGradient',
  full_name='tensorflow.RegisteredGradient',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gradient_func', full_name='tensorflow.RegisteredGradient.gradient_func', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='registered_op_type', full_name='tensorflow.RegisteredGradient.registered_op_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=1149,
  serialized_end=1220,
)

_FUNCTIONDEFLIBRARY.fields_by_name['function'].message_type = _FUNCTIONDEF
_FUNCTIONDEFLIBRARY.fields_by_name['gradient'].message_type = _GRADIENTDEF
_FUNCTIONDEFLIBRARY.fields_by_name['registered_gradients'].message_type = _REGISTEREDGRADIENT
_FUNCTIONDEF_ATTRENTRY.fields_by_name['value'].message_type = attr__value__pb2._ATTRVALUE
_FUNCTIONDEF_ATTRENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_ARGATTRS_ATTRENTRY.fields_by_name['value'].message_type = attr__value__pb2._ATTRVALUE
_FUNCTIONDEF_ARGATTRS_ATTRENTRY.containing_type = _FUNCTIONDEF_ARGATTRS
_FUNCTIONDEF_ARGATTRS.fields_by_name['attr'].message_type = _FUNCTIONDEF_ARGATTRS_ATTRENTRY
_FUNCTIONDEF_ARGATTRS.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_ARGATTRENTRY.fields_by_name['value'].message_type = _FUNCTIONDEF_ARGATTRS
_FUNCTIONDEF_ARGATTRENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_RETENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF_CONTROLRETENTRY.containing_type = _FUNCTIONDEF
_FUNCTIONDEF.fields_by_name['signature'].message_type = op__def__pb2._OPDEF
_FUNCTIONDEF.fields_by_name['attr'].message_type = _FUNCTIONDEF_ATTRENTRY
_FUNCTIONDEF.fields_by_name['arg_attr'].message_type = _FUNCTIONDEF_ARGATTRENTRY
_FUNCTIONDEF.fields_by_name['resource_arg_unique_id'].message_type = _FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY
_FUNCTIONDEF.fields_by_name['node_def'].message_type = node__def__pb2._NODEDEF
_FUNCTIONDEF.fields_by_name['ret'].message_type = _FUNCTIONDEF_RETENTRY
_FUNCTIONDEF.fields_by_name['control_ret'].message_type = _FUNCTIONDEF_CONTROLRETENTRY
DESCRIPTOR.message_types_by_name['FunctionDefLibrary'] = _FUNCTIONDEFLIBRARY
DESCRIPTOR.message_types_by_name['FunctionDef'] = _FUNCTIONDEF
DESCRIPTOR.message_types_by_name['GradientDef'] = _GRADIENTDEF
DESCRIPTOR.message_types_by_name['RegisteredGradient'] = _REGISTEREDGRADIENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FunctionDefLibrary = _reflection.GeneratedProtocolMessageType('FunctionDefLibrary', (_message.Message,), dict(
  DESCRIPTOR = _FUNCTIONDEFLIBRARY,
  __module__ = 'function_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.FunctionDefLibrary)
  ))
_sym_db.RegisterMessage(FunctionDefLibrary)

FunctionDef = _reflection.GeneratedProtocolMessageType('FunctionDef', (_message.Message,), dict(

  AttrEntry = _reflection.GeneratedProtocolMessageType('AttrEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_ATTRENTRY,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.AttrEntry)
    ))
  ,

  ArgAttrs = _reflection.GeneratedProtocolMessageType('ArgAttrs', (_message.Message,), dict(

    AttrEntry = _reflection.GeneratedProtocolMessageType('AttrEntry', (_message.Message,), dict(
      DESCRIPTOR = _FUNCTIONDEF_ARGATTRS_ATTRENTRY,
      __module__ = 'function_pb2'
      # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.ArgAttrs.AttrEntry)
      ))
    ,
    DESCRIPTOR = _FUNCTIONDEF_ARGATTRS,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.ArgAttrs)
    ))
  ,

  ArgAttrEntry = _reflection.GeneratedProtocolMessageType('ArgAttrEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_ARGATTRENTRY,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.ArgAttrEntry)
    ))
  ,

  ResourceArgUniqueIdEntry = _reflection.GeneratedProtocolMessageType('ResourceArgUniqueIdEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.ResourceArgUniqueIdEntry)
    ))
  ,

  RetEntry = _reflection.GeneratedProtocolMessageType('RetEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_RETENTRY,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.RetEntry)
    ))
  ,

  ControlRetEntry = _reflection.GeneratedProtocolMessageType('ControlRetEntry', (_message.Message,), dict(
    DESCRIPTOR = _FUNCTIONDEF_CONTROLRETENTRY,
    __module__ = 'function_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef.ControlRetEntry)
    ))
  ,
  DESCRIPTOR = _FUNCTIONDEF,
  __module__ = 'function_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.FunctionDef)
  ))
_sym_db.RegisterMessage(FunctionDef)
_sym_db.RegisterMessage(FunctionDef.AttrEntry)
_sym_db.RegisterMessage(FunctionDef.ArgAttrs)
_sym_db.RegisterMessage(FunctionDef.ArgAttrs.AttrEntry)
_sym_db.RegisterMessage(FunctionDef.ArgAttrEntry)
_sym_db.RegisterMessage(FunctionDef.ResourceArgUniqueIdEntry)
_sym_db.RegisterMessage(FunctionDef.RetEntry)
_sym_db.RegisterMessage(FunctionDef.ControlRetEntry)

GradientDef = _reflection.GeneratedProtocolMessageType('GradientDef', (_message.Message,), dict(
  DESCRIPTOR = _GRADIENTDEF,
  __module__ = 'function_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.GradientDef)
  ))
_sym_db.RegisterMessage(GradientDef)

RegisteredGradient = _reflection.GeneratedProtocolMessageType('RegisteredGradient', (_message.Message,), dict(
  DESCRIPTOR = _REGISTEREDGRADIENT,
  __module__ = 'function_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.RegisteredGradient)
  ))
_sym_db.RegisterMessage(RegisteredGradient)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\016FunctionProtosP\001ZOgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/function_go_proto\370\001\001'))
_FUNCTIONDEF_ATTRENTRY.has_options = True
_FUNCTIONDEF_ATTRENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_ARGATTRS_ATTRENTRY.has_options = True
_FUNCTIONDEF_ARGATTRS_ATTRENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_ARGATTRENTRY.has_options = True
_FUNCTIONDEF_ARGATTRENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY.has_options = True
_FUNCTIONDEF_RESOURCEARGUNIQUEIDENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_RETENTRY.has_options = True
_FUNCTIONDEF_RETENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
_FUNCTIONDEF_CONTROLRETENTRY.has_options = True
_FUNCTIONDEF_CONTROLRETENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
