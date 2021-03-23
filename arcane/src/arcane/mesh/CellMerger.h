// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellMerger.h                                                (C) 2000-2020 */
/*                                                                           */
/* Fusionne deux mailles.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CELLMERGER_H
#define ARCANE_MESH_CELLMERGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemInternal;
}

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe d'aide pour fusionner deux mailles.
 */
class ARCANE_MESH_EXPORT CellMerger
{
 public:

  //! Constructeur
  CellMerger(ITraceMng* tm) : m_trace_mng(tm) { }

  //! Destructeur
  ~CellMerger() = default;

 public:

  /*!
   * \brief Effectue la fusion des deux mailles \a i_cell_1 et \a i_cell_2.
   * 
   * \param i_cell_1 un pointeur sur la premiere maille
   * \param i_cell_2 un pointeur sur la deuxième maille
   * 
   * \note la fusion est \b toujours effectuée dans \a i_cell_1, \a i_cell_2
   * devient une maille applatie qui sera détruite par la suite.
   */
  void merge(ItemInternal* i_cell_1,ItemInternal* i_cell_2);

  /*!
   * \brief Retourne l'ItemInteral utilisé par la maille après fusion
   * 
   * \param i_cell_1 un pointeur sur la premiere maille
   * \param i_cell_2 un pointeur sur la deuxième maille
   * 
   * \return un pointeur sur la nouvelle maille.
   * 
   * \note le nouveau pointeur est toujours soit \a i_cell_1 soit
   * \a i_cell_2. Aucune allocation de mémoire n'est effectuée.
   */
  ItemInternal* getItemInternal(ItemInternal* i_cell_1, ItemInternal* i_cell_2);

 private:

  ITraceMng* m_trace_mng;

 private:

 /*!
   * On se donne un type énuméré local afin de pouvoir effectuer des
   * operations arithmétiques (voir \see _promoteType)
   */
  enum _Type
  {
    NotMergeable  = 0,
    Hexahedron    = 1,
    Pyramid       = 2,
    Pentahedron   = 3,
    Quadrilateral = 10,
    Triangle      = 11
  };

  /*!
   * \brief Retourne le nom associé à type de maille
   * 
   * \param t le type
   * 
   * \return la chaîne contenant le nom.
   */
  String _typeName(const _Type& t) const;

  /*!
   * \brief Détermine le _Type de la maille en fonction de son type "ItemInternal"
   * 
   * \param internal_cell_type le type "ItemInternal"
   * 
   * \return le _Type de la maille
   */
  _Type _getCellType(const Integer& internal_cell_type) const;

  /*!
   * \bfrief Détermine le type de maille résultat de la fusion de deux types donnés.
   * \note à ce stade rien ne garanti que la fusion va aboutir
   * 
   * \param t1 le 1er type
   * \param t2 le second type
   * 
   * \return le type de la maille fusionnée
   */
  _Type _promoteType(const _Type& t1, const _Type& t2) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // CELL_MERGER_H
