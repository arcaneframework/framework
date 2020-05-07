//
// Created by dechaiss on 5/6/20.
//

#ifndef NEO_MESH_H
#define NEO_MESH_H

/*-------------------------
 * sdc - (C) 2019-2020
 * NEtwork Oriented kernel
 * POC Mesh API
 *--------------------------
 */

#include <memory>

namespace Neo {

class MeshBase;

class Mesh {

  Mesh(const std::string& mesh_name);
  ~Mesh();

private:

  std::unique_ptr<MeshBase> mesh_graph;
};

} // end namespace Neo

#endif // NEO_MESH_H
