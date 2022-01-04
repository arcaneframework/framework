// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemRefinement.h                                            (C) 2000-2010 */
/*  Created on: Jul 16, 2010                                                 */
/*      Author: mesriy                                                       */
/*                                                                           */
/* méthodes de manipulation d'un Item du maillage par des techniques AMR.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMREFINEMENT_H
#define ARCANE_MESH_ITEMREFINEMENT_H

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/MeshRefinement.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR

#include<set>
#include<vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class IMesh;


ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemRefinement
: public TraceAccessor
{
public:
	/**
	 * Constructor.
	 */
	ItemRefinement (IMesh *mesh);

private:
	// le ctor de copie et l'opérateur d'affectation sont
	// déclarés privés mais non implémentés.  C'est la
	// technique standard pour les empêcher d'être utilisés.
	ItemRefinement (const ItemRefinement&);
	ItemRefinement& operator=(const ItemRefinement&);

public:

	/**
	 * Destructor.
	 */
	~ItemRefinement();

	struct FaceSetCompare
	{
		typedef std::set<Int64> FaceSet;
		bool operator()(const FaceSet& s1, const FaceSet& s2) const
		{
			//return (s1 == s2);
				if (s1.size() != s2.size())
					return false;
				else {
					FaceSet::const_iterator its1 = s1.begin(), its2 = s2.begin();
					FaceSet::const_iterator its1_end = s1.end();
					for(; its1 != its1_end; ++its1,++its2)
						if (*its1 != *its2)
							return false;
				}
			    return true;
		}
	};

	//!
	template <int typeID>
	void refineOneCell(Cell item, MeshRefinement& mesh_refinement);

	//!
  template <int typeID>
  void coarsenOneCell(Cell item, const ItemRefinementPatternT<typeID>& rp);

  void initHMin() ;
  void updateChildHMin(ArrayView<ItemInternal*> cells) ;

 private:
	Real hmin(Cell) const;
	Real3 faceCenter(ItemInternal* face,SharedVariableNodeReal3& nodes_coords) const;

	template <int typeID>
	void computeHChildren(Cell item, MeshRefinement& mesh_refinement);

	template <int typeID>
	void computeOrigNodesCoords(Cell item, const ItemRefinementPatternT<typeID>& rp, const Integer sid);

private:
	IMesh* m_mesh;
	VariableCellReal m_cell_hmin;
  VariableNodeReal3& m_orig_nodes_coords;

	Integer m_refine_factor;
	Integer m_nb_cell_to_add;
	Integer m_nb_face_to_add;
	Integer m_nb_node_to_add;

	// hmin*TOLERENCE= tolerence de recherche d'un nouveau point.
	static const Real TOLERENCE;

  // Reel3 m_p[c,nc]  : coordinates of node nc of children c
  std::vector<std::vector<Real3> > m_p;
  // Int64 m_nodes_uid[c,nc]  : uid of node nc of children c
  std::vector<std::vector<Int64> > m_nodes_uid;
  Real3UniqueArray m_coord;
  Int64UniqueArray m_cells_infos;
  Int64UniqueArray m_faces_infos;
  Int64UniqueArray m_face;
  Real3UniqueArray m_nodes_to_create_coords;
  Int64UniqueArray m_nodes_unique_id;
  Int32UniqueArray m_nodes_lid;
  Int32UniqueArray m_faces_lid;
  Int32UniqueArray m_cells_lid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_MESH_ITEMREFINEMENT_H */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
