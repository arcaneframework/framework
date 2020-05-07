//
// Created by dechaiss on 5/6/20.
//

#include "neo/Mesh.h"
#include "neo/Neo.h"


Neo::Mesh::Mesh(const std::string& mesh_name)
 : mesh_graph(std::make_unique<Neo::MeshBase>(Neo::MeshBase{mesh_name})){
}

Neo::Mesh::~Mesh() = default;

std::string const& Neo::Mesh::name() const {
  return mesh_graph->m_name;
}
