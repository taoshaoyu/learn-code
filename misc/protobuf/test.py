import graph_pb2

graph = graph_pb2.GraphDef()
fn='/home/taosy/work/opensrc/github.com/shaoyuta/learn-code/ssd_resnet34_fp32_1200x1200_pretrained_model.pb'

with open(fn,"rb") as f:
    graph.ParseFromString(f.read())

for node in graph.node:
  print(f'{node.input} -> {node.name}')