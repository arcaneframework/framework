//
// Created by dechaiss on 5/6/20.
//

#include "neo/Mesh.h"
#include "neo/Neo.h"


Neo::Mesh::Mesh(const std::string& mesh_name)
 : m_mesh_graph(std::make_unique<Neo::MeshBase>(Neo::MeshBase{mesh_name})){
}

Neo::Mesh::~Mesh() = default;

std::string const& Neo::Mesh::name() const noexcept {
  return m_mesh_graph->m_name;
}

Neo::Family& Neo::Mesh::addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept
{
  auto& cell_family = m_mesh_graph->addFamily(item_kind, std::move(family_name));
  cell_family.addProperty<Neo::utils::Int64>(family_name + "_uids");
  return cell_family;
}
