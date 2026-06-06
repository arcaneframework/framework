// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleSVGMeshExporter.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Mesh exporter in SVG format.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Must put this first for MSVC otherwise we don't have 'M_PI'
#define _USE_MATH_DEFINES
#include <cmath>

#include "arcane/SimpleSVGMeshExporter.h"

#include "arcane/utils/Iostream.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/VariableTypes.h"

#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using std::ostream;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleSVGMeshExporter::Impl
{
 public:
  Impl(ostream& ofile) : m_ofile(ofile){}
  void _writeText(Real x,Real y,StringView color,StringView text,double rotation,bool do_background);
  void write(const CellGroup& cells);
 private:
  ostream& m_ofile;
  double m_font_size = 3.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleSVGMeshExporter::Impl::
_writeText(Real x,Real y,StringView color,StringView text,double rotation,bool do_background)
{
  // Displays a white background beneath the text.
  if (do_background){
    m_ofile << "<text x='" << x << "' y='" << y << "' dominant-baseline='central' text-anchor='middle'"
            << " style='stroke:white; stroke-width:0.6em'";
    if (rotation!=0.0)
      m_ofile << " transform='rotate(" << rotation << "," << x << "," << y << ")'";
    m_ofile << " font-size='" << m_font_size << "'>" << text << "</text>\n";
  }
  m_ofile << "<text x='" << x << "' y='" << y << "' dominant-baseline='central' text-anchor='middle'"
          << " fill='" << color << "'";
  if (rotation!=0.0)
    m_ofile << " transform='rotate(" << rotation << "," << x << "," << y << ")'";
  m_ofile << " font-size='" << m_font_size << "'>" << text << "</text>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleSVGMeshExporter::Impl::
write(const CellGroup& cells)
{
  if (cells.null())
    return;
  IMesh* mesh = cells.mesh();
  Int32 mesh_dim = mesh->dimension();
  if (mesh_dim != 2)
    ARCANE_FATAL("Invalid dimension ({0}) for mesh. Only 2D mesh is allowed", mesh_dim);

  // Note: since the origin in SVG is by default top-left, we take the opposite of each
  // 'Y' value for display.
  // NOTE: we could use SVG transformations, but that is more complicated to
  // handle for text display.

  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  ostream& ofile = m_ofile;
  Real mul_value = 1000.0;
  const Real min_val = std::numeric_limits<Real>::lowest();
  const Real max_val = std::numeric_limits<Real>::max();
  Real2 min_bbox(max_val, max_val);
  Real2 max_bbox(min_val, min_val);
  // Calculates the center of the meshes and the bounding box of the mesh group.
  std::map<Int32, Real2> cells_center;
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Real3 center;
    Integer nb_node = cell.nbNode();
    if (nb_node > 0) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real3 node_coord_3d = nodes_coord[cell.node(i)] * mul_value;
        Real2 node_coord(node_coord_3d.x, -node_coord_3d.y);
        min_bbox = math::min(min_bbox, node_coord);
        max_bbox = math::max(max_bbox, node_coord);
        center += node_coord_3d;
      }
      center /= nb_node;
    }
    Real2 center_2d(center.x, -center.y);
    cells_center[cell.localId()] = center_2d;
  }

  Real bbox_width = math::abs(max_bbox.x - min_bbox.x);
  Real bbox_height = math::abs(max_bbox.y - min_bbox.y);
  Real max_dim = math::max(bbox_width, bbox_height);
  m_font_size = max_dim / 80.0;

  // Adds 10% of the dimensions on both sides of the viewBox to ensure
  // that the text is properly written (as it might overflow the bounding box)
  ofile << "<?xml version=\"1.0\"?>\n";
  ofile << "<svg"
        << " viewBox='" << min_bbox.x - bbox_width * 0.1 << "," << min_bbox.y - bbox_height * 0.1 << "," << bbox_width * 1.2 << "," << bbox_height * 1.2 << "'"
        << " xmlns='http://www.w3.org/2000/svg' version='1.1'>\n";
  ofile << "<!-- V3 bbox min_x=" << min_bbox.x << " min_y=" << min_bbox.y << " max_x=" << max_bbox.x << " max_y=" << max_bbox.y << " -->";
  ofile << "<title>Mesh</title>\n";
  ofile << "<desc>MeshExample</desc>\n";

  //ofile << "<g transform='matrix(1,0,0,-1,0,200)'>\n";
  //ofile << "<g transform='translate(" << min_bbox.x << "," << -min_bbox.y << ")'>\n";
  ofile << "<g>\n";

  // Displays the contour and uniqueId() for each mesh.
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    Real2 cell_pos = cells_center[cell.localId()];
    Integer nb_node = cell.nbNode();
    ofile << "<path d='";
    nb_node = cell.typeInfo()->linearTypeInfo()->nbLocalNode();
    for (Integer i = 0; i < nb_node; ++i) {
      Real3 node_coord_3d = nodes_coord[cell.node(i)];
      Real2 node_coord(node_coord_3d.x, -node_coord_3d.y);
      node_coord *= mul_value;
      if (i == 0)
        ofile << "M ";
      else
        ofile << "L ";
      // performs a homothety to clearly see the faces in case of welding.
      Real2 coord = cell_pos + (node_coord - cell_pos) * 0.98;
      ofile << coord.x << " " << coord.y << " ";
    }
    ofile << "z'";
    if (cell.isOwn())
      ofile << " fill='yellow'";
    else
      ofile << " fill='orange'";
    ofile << " stroke='black'";
    ofile << " stroke-width='1'/>\n";
    _writeText(cell_pos.x, cell_pos.y, "blue", String::fromNumber(cell.uniqueId().asInt64()), 0.0, false);
  }

  // Displays the uniqueId() for each node.
  {
    // Set of nodes already processed to display them only once.
    std::set<Int32> nodes_done;
    ENUMERATE_CELL (icell, cells) {
      Cell cell = *icell;
      Integer nb_node = cell.nbNode();
      for (Integer i = 0; i < nb_node; ++i) {
        Node node = cell.node(i);
        Int32 lid = node.localId();
        if (nodes_done.find(lid) != nodes_done.end())
          continue;
        nodes_done.insert(lid);
        Real3 coord_3d = nodes_coord[node];
        Real2 coord(coord_3d.x, -coord_3d.y);
        coord *= mul_value;
        _writeText(coord.x, coord.y, "green", String::fromNumber(node.uniqueId().asInt64()), 0.0, true);
      }
    }
  }

  // Displays the uniqueId() for each face.
  // Performs a possible rotation so that the display of the face number is aligned
  // with its segment.
  {
    // Set of faces already processed to display them only once.
    std::set<Int32> faces_done;
    ENUMERATE_CELL (icell, cells) {
      Cell cell = *icell;
      Integer nb_face = cell.nbFace();
      for (Integer i = 0; i < nb_face; ++i) {
        Face face = cell.face(i);
        Int32 lid = face.localId();
        if (faces_done.find(lid) != faces_done.end())
          continue;
        faces_done.insert(lid);
        // In the case of multi-dimensional meshing, it is possible
        // to have faces reduced to a point.
        if (face.nbNode()<2)
          continue;
        Real3 node0_coord = nodes_coord[face.node(0)];
        Real3 node1_coord = nodes_coord[face.node(1)];
        Real3 face_coord_3d = (node0_coord + node1_coord) / 2.0;

        Real2 face_coord(face_coord_3d.x, -face_coord_3d.y);
        face_coord *= mul_value;
        Real3 direction = node1_coord - node0_coord;
        direction = direction.normalize();
        // TODO: check between -1.0 and 1.0
        // Calculates the rotation angle so that the display of the face number is aligned with
        // the edge of the face.
        double angle = math::abs(std::asin(direction.y)) / M_PI * 180.0;
        Real2 cell_center = cells_center[cell.localId()];
        Real2 coord = cell_center + (face_coord - cell_center) * 0.92;
        _writeText(coord.x, coord.y, "red", String::fromNumber(face.uniqueId().asInt64()), angle, true);
      }
    }
  }

  ofile << "</g>\n";
  ofile << "</svg>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleSVGMeshExporter::
SimpleSVGMeshExporter(std::ostream& ofile)
: m_p(new Impl(ofile))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleSVGMeshExporter::
~SimpleSVGMeshExporter()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleSVGMeshExporter::
write(const CellGroup& cells)
{
  m_p->write(cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
