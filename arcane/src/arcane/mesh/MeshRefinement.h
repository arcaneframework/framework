// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshRefinement.h                                            (C) 2000-2024 */
/*                                                                           */
/* Gestion de l'adaptation de maillages non-structurés par raffinement       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHREFINEMENT_H
#define ARCANE_MESH_MESHREFINEMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/PerfCounterMng.h"
#include "arcane/utils/AMRCallBackMng.h"

#include "arcane/core/ItemRefinementPattern.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/MapCoordToUid.h"
#include "arcane/mesh/DynamicMeshKindInfos.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMesh;
class Node;
class Cell;
class AMRCallBackMng;
class FaceFamily;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

class DynamicMesh;
class ItemRefinement;
class ParallelAMRConsistency;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des algorithmes d'adaptation par raffinement de
 * maillages non-structuré.
 */
class MeshRefinement
:  public TraceAccessor
{
public:
#ifdef ACTIVATE_PERF_COUNTER
  struct PerfCounter
  {
      typedef enum {
        INIT,
        CLEAR,
        ENDUPDATE,
        UPDATEMAP,
        UPDATEMAP2,
        CONSIST,
        PCONSIST,
        PCONSIST2,
        PGCONSIST,
        CONTRACT,
        COARSEN,
        REFINE,
        INTERP,
        PGHOST,
        COMPACT,
        NbCounters
      }  eType ;

      static const std::string m_names[NbCounters] ;
  } ;
#endif
  /**
   * Constructor.
   */
  MeshRefinement(DynamicMesh* mesh);

private:
  // le ctor de copie et l'opérateur d'affectation sont
  // déclarés privés mais non implémentés.  C'est la
  // technique standard pour les empêcher d'être utilisés.
  MeshRefinement(const MeshRefinement&);
  MeshRefinement& operator=(const MeshRefinement&);

public:

  /**
   * Destructor.
   */
  ~MeshRefinement();

  /**
   * Supprime toutes les données qui sont actuellement stockés.
   */
  void clear();
  /**
   * Calcul du max des uid.
   */
  void init();

  void update() ;
  bool needUpdate() const {
    return m_need_update ;
  }
  void invalidate() {
    m_need_update = true ;
  }
  void initMeshContainingBox() ;
  /**
   * Flag les items pour raffinement/déraffinement
   */
  void flagItems(const Int32Array& flag_per_cell,
                 const Integer max_level = -1);

  /*!
   * \brief Passage de l'erreur commise par maille au flag de raffinement.
   *
   * Cette métthode pourrait être condée de manières différentes:
   * 1- implémentation actuelle: l'uilisateur fait la transformation lui-même
   * dans ce cas, il modifie l'objet itemInternal en settant le flag de raffinement
   * 2- l'uilisateur fait la transformation lui-même mais stocke et retourne un tableau des flags
   * la classe MeshRefinement, dans ce cas là, implémente un setter à partir du tableau retourné ici
   * 3- pour éviter la copie du tableau des flags, implémenter le converter directement dans meshRefinement
   * et l'utilisateur ne fait que fournir le tableau d'erreur
   */
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids);
  virtual void flagCellToCoarsen(Int32ConstArrayView cells_lids);

  /*!
   * Raffine et déraffine les items demandés par l'utilisateur. également
   * raffine/déraffine des items complémentaires pour satisfaire la règle de niveau-un.
   * Il est possible que pour un ensemble donné de flags qu'il
   * n'est réellement aucun changement en appelant cette méthode. En conséquence,
   * elle envoie \p true si le maillage a changé réellement (dans ce cas
   * les données doivent être projetées) et \p false sinon.

   * L'argument \p maintain_level_one est deprecated; utilisez plutôtt l'option
   * face_level_mismatch_limit() .
   */
  bool refineAndCoarsenItems(const bool maintain_level_one=true);

  /*!
   * Dérffine seulement les items demandés par l'utilisateur. Quelques items
   * ne seront pas déraffinés pour satisfaire la régle de niveau un.
   * Il est possible que pour un ensemble donné de flags qu'il
   * n'est réellement aucun changement en appelant cette méthode.  En conséquence,
   * elle envoie \p true si le maillage a changé réellement (dans ce cas
   * les données doivent être projetées) et \p false sinon.

   * L'argument \p maintain_level_one est deprecated; utilisez plutôt l'option
   * face_level_mismatch_limit() .
   */
  bool coarsenItems(const bool maintain_level_one = true);

  bool coarsenItemsV2();

  /*!
   * raffine seulement les items demandés par l'utilisateur.
   * Il est possible que pour un ensemble donné de flags qu'il
   * n'est réellement aucun changement en appelant cette méthode.  En conséquence,
   * elle envoie \p true si le maillage a changé réellement (dans ce cas
   * les données doivent être projetées) et \p false sinon.

   * L'argument \p maintain_level_one est deprecated; utilisez plutôtt l'option
   * face_level_mismatch_limit() .
   */
  bool refineItems(const bool maintain_level_one=true);

  /*!
   * Raffine uniformement le maillage \p n fois.
   */
  void uniformlyRefine(Integer n=1);

  /*!
   * déraffine uniformement le maillage \p n fois.
   */
  void uniformlyCoarsen(Integer n=1);

  /*!
   * \p max_level est le plus grand niveau de raffinement
   * qu'un item peut atteindre.
   *
   * \p max_level est illimité (-1) par default
   */
  Integer& maxLevel();

  //! Référence constante au maillage.
  const IMesh* getMesh() const;

  //! Référence au maillage.
  IMesh* getMesh();

  //!
  void registerCallBack(IAMRTransportFunctor* f);
  //!
  void unRegisterCallBack(IAMRTransportFunctor* f);
  /*!
   * Ajout d'un nouveau uid associé au point \p p.
   * si p existe déjà, on garde l'ancien uid.
   * La tolerance \p tol donne le perimetre de recherche autour de p.
   */
  Int64 findOrAddNodeUid(const Real3& p,const Real& tol);
  /*!
   * Ajout d'un nouveau uid associé au centre de la face \p face_center.
   * si p existe déjà, on garde l'ancien uid.
   * La tolerance \p tol donne le perimetre de recherche autour de face_center.
   */
  Int64 findOrAddFaceUid(const Real3& face_center,const Real& tol,bool& is_added);
  /*!
   * genere un nouveau uid pour les enfants.
   */
  Int64 getFirstChildNewUid();

  void _update(ArrayView<ItemInternal*> cells_to_refine);
  void _update(ArrayView<Int64> cells_to_refine);
  void _invalidate(ArrayView<ItemInternal*> cells_to_coarsen);
  void _updateMaxUid(ArrayView<ItemInternal*> cells_to_refine);

  /*!
   * return le pattern de raffinement associe au type de maille.
   */
  template <int typeID> const ItemRefinementPatternT<typeID>& getRefinementPattern() const ;
  /*!
   * Determination des connexions non conformes des mailles raffinees.
   */
  void populateBackFrontCellsFromChildrenFaces(Cell parent_cell);
  void populateBackFrontCellsFromParentFaces(Cell parent_cell);

 private:

  /*!
   * Retourne true si et seulement si le maillage satisfait la règle de niveau un
   * Retourne false sinon
   * Arrète l'exècution si arcane_assert_yes est true et si
   * le maillage ne satisfait pas la règle de niveau un
   */
  bool _checkLevelOne(bool arcane_assert_yes = false);

  /*!
   * Retourne true si et seulement si le maillage n'a pas d'items
   * flaggès pour déraffinement ou raffinement
   * Retourne false autrement
   * Arrète l'exécution si \a arcane_assert_yes est true et si
   * le maillage a des items flaggés
   */
  bool _checkUnflagged(bool arcane_assert_yes = false);

  /*!
   * Si \p coarsen_by_parents est true,
   * les items avec le même parent seront flaggés pour déraffinement
   * Ceci devrait produire un déraffinement plus proche à ce qui a été demandé.
   *
   * \p coarsen_by_parents est true par default.
   */
  bool& coarsenByParents();


  /*!
   * Si Face_level_mismatch_limit est mise à une valeur non nulle, alors
   * le raffinement et déraffinement produiront des maillages dans lesquels
   * le niveau de raffinement de deux mailles voisines par face ne différera pas plus que
   * cette limite.  Si Face_level_mismatch_limit est 0, donc les diffèrences de niveau
   * seront illimitées.
   *
   * \p face_level_mismatch_limit est 1 par default.  Actuellement les seules
   * options supportèes sont 0 et 1.
   */
  unsigned char& faceLevelMismatchLimit();

  /*!
   * Supprime les enfants subactifs du maillage
   * Contracte un item actif, i.e. supprimer les pointers vers chaque
   * enfant subactif.  Ceci devrait seulement ètre appelè après restriction des variables
   * sur les parents
   */
  bool _contract();

  //! interpolation des données sur les mailles enfants
  void _interpolateData(const Int64Array& cells_to_refine);

  //! restriction des données sur les mailles parents
  void _upscaleData(Array<ItemInternal*>& parent_cells);

 private:

  /**
   * Déraffine les items demandés par l'utilisateur. Les deux méthodes _coarsenItems()
   * et _refineItems() ne sont pas dans l'interface publique de MeshRefinement. Car une
   * prèparation approprièe (makeRefinementCompatible, makeCoarseningCompatible) est
   * nècessaire pour qu'on puisse exècuter _coarsenItems().
   *
   * Il est possible que pour un ensemble donné de flags qu'il
   * n'est réellement aucun changement en appelant cette fonction.  En conséquence,
   * Elle renvoie \p true si le maillage a changé réellement (dans ce cas
   * les données doivent ètre projetées) \p false autrement.
   */
  bool _coarsenItems();

  /**
   * Raffine les items demandès par l'utilisateur.
   *
   * Il est possible que pour un ensemble donné de flags qu'il
   * n'est rèellement aucun changement en appelant cette fonction.  En conséquence,
   * Elle renvoie \p true si le maillage a changé réellement (dans ce cas
   * les données doivent ètre projetées) \p false autrement.
   */
  bool _refineItems(Int64Array& cells_to_refine);

  // mise a jour des owner des items a partir des mailles
  void _updateItemOwner(Int32ArrayView cell_to_remove);
  void _updateItemOwner2();
  //!
  bool _removeGhostChildren();
  //---------------------------------------------
  // Utility algorithms


  /**
   * Mise è jour de m_nodes_finder et m_faces_finder
   */
  void _updateLocalityMap();
  void _updateLocalityMap2();
  /**
   * Mettre le flag de raffinement à II_DoNothing
   * pour chaque item du maillage.
   */
  void _cleanRefinementFlags();

  /**
   * Agit sur les flags de déraffinement à ce que la règle
   * de niveau-un soit satisaite.
   */
  bool _makeCoarseningCompatible(const bool);


  /**
   * Agir sur les flags de raffinement à ce que la règle
   * de niveau-un soit satisaite.
   */
  bool _makeRefinementCompatible(const bool);

  /**
   * Copie des flags de raffinement sur les items frontières à partir de leur
   * processeurs owners.  Retourne true si un flag a changè.
   */
  bool _makeFlagParallelConsistent();
  bool _makeFlagParallelConsistent2();

  /**
   * Determination des connexions non conformes des mailles raffinées
   */
  template <int typeID>
  void _populateBackFrontCellsFromParentFaces(Cell parent_cell) ;
  template <int typeID>
  void _populateBackFrontCellsFromChildrenFaces(Face face,Cell parent_cell,
                                                Cell neighbor_cell);

  void _checkOwner(const String & msg); // To avoid owner desynchronization

 private:

  /**
   * Reference au maillage.
   */
  DynamicMesh* m_mesh;
  FaceFamily* m_face_family;
  bool m_need_update;

  /**
   * recherche rapide des noeuds et des faces a partir de leurs coords
   * pour les faces, les coords sont celles du centre de la face
   */
  MapCoordToUid::Box m_mesh_containing_box ;
  NodeMapCoordToUid m_node_finder;
  FaceMapCoordToUid m_face_finder;

  /**
   * reference au raffineur d'items
   */
  ItemRefinement * m_item_refinement;

  /**
   * assure la consistence des uid en parallèle
   */
  ParallelAMRConsistency* m_parallel_amr_consistency;

  /**
   * gestionnaire des functors de transport de donnees entre maillages
   */
  AMRCallBackMng* m_call_back_mng;

  /**
   * paramètres de Raffinement
   */

  bool m_coarsen_by_parents;
  Integer m_max_level;
  Integer m_nb_cell_target;
  Byte m_face_level_mismatch_limit;

  Int64 m_max_node_uid;
  Int64 m_next_node_uid;

  Int64 m_max_cell_uid;
  Int64 m_next_cell_uid;

  Int64 m_max_face_uid;
  Int64 m_next_face_uid;

  Integer m_max_nb_hChildren;

  /**
   * pattern de Raffinement
   */
  const Quad4RefinementPattern4Quad m_quad4_refinement_pattern;
  const TetraRefinementPattern2Hex_2Penta_2Py_2Tetra m_tetra_refinement_pattern;
  const PyramidRefinementPattern4Hex_4Py m_pyramid_refinement_pattern;
  const PrismRefinementPattern4Hex_4Pr m_prism_refinement_pattern;
  const HexRefinementPattern8Hex m_hex_refinement_pattern;
  const HemiHex7RefinementPattern6Hex_2HHex7 m_hemihexa7_refinement_pattern;
  const HemiHex6RefinementPattern4Hex_4HHex7 m_hemihexa6_refinement_pattern;
  const HemiHex5RefinementPattern2Hex_4Penta_2HHex5 m_hemihexa5_refinement_pattern;
  const AntiWedgeLeft6RefinementPattern4Hex_4HHex7 m_antiwedgeleft6_refinement_pattern;
  const AntiWedgeRight6RefinementPattern4Hex_4HHex7 m_antiwedgeright6_refinement_pattern;
  const DiTetra5RefinementPattern2Hex_6HHex7 m_ditetra5_refinement_pattern;

  // Node owner memory : to patch owner desynchronization
  VariableNodeInt32 m_node_owner_memory;

#ifdef ACTIVATE_PERF_COUNTER
  PerfCounterMng<PerfCounter> m_perf_counter ;
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// MeshRefinement class inline members

inline bool& MeshRefinement::coarsenByParents()
{
  return m_coarsen_by_parents;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer& MeshRefinement::maxLevel()
{
  return m_max_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline unsigned char& MeshRefinement::faceLevelMismatchLimit()
{
  return m_face_level_mismatch_limit;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
inline const ItemRefinementPatternT<IT_Quad4>&
MeshRefinement::getRefinementPattern<IT_Quad4>() const
{
  return m_quad4_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Tetraedron4>&
MeshRefinement::getRefinementPattern<IT_Tetraedron4>() const
{
  return m_tetra_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Pyramid5>&
MeshRefinement::getRefinementPattern<IT_Pyramid5>() const
{
  return m_pyramid_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Pentaedron6>&
MeshRefinement::getRefinementPattern<IT_Pentaedron6>() const
{
  return m_prism_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_Hexaedron8>&
MeshRefinement::getRefinementPattern<IT_Hexaedron8>() const
{
  return m_hex_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa7>&
MeshRefinement::getRefinementPattern<IT_HemiHexa7>() const
{
  return m_hemihexa7_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa6>&
MeshRefinement::getRefinementPattern<IT_HemiHexa6>() const
{
  return m_hemihexa6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_HemiHexa5>&
MeshRefinement::getRefinementPattern<IT_HemiHexa5>() const
{
  return m_hemihexa5_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_AntiWedgeLeft6>&
MeshRefinement::getRefinementPattern<IT_AntiWedgeLeft6>() const
{
  return m_antiwedgeleft6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_AntiWedgeRight6>&
MeshRefinement::getRefinementPattern<IT_AntiWedgeRight6>() const
{
  return m_antiwedgeright6_refinement_pattern;
}
template <>
inline const ItemRefinementPatternT<IT_DiTetra5>&
MeshRefinement::getRefinementPattern<IT_DiTetra5>() const
{
  return m_ditetra5_refinement_pattern;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // end ARCANE_MESH_MESHREFINEMENT_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
