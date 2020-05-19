//
// Created by dechaiss on 5/6/20.
//

#ifndef NEO_MESH_H
#define NEO_MESH_H

/*-------------------------
 * sdc - (C) 2020
 * NEtwork Oriented kernel
 * POC Mesh API
 *--------------------------
 */

#include <memory>
#include <string>
#include <vector>

#include "Neo.h"
#include "neo/Utils.h"

namespace Neo {

class MeshBase;
enum class ItemKind;
class Family;

class Mesh {

public:
  Mesh(std::string const& mesh_name);
  ~Mesh();

private:
  std::unique_ptr<MeshBase> m_mesh_graph;

public:
  std::string const& name() const noexcept ;

  Neo::Family&  addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept ;
};

} // end namespace Neo

#endif // NEO_MESH_H
