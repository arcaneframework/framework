// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemIndexArrayView.h                                        (C) 2000-2023 */
/*                                                                           */
/* Vue sur un tableau d'index (localIds()) d'entités.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINDEXARRAYVIEW_H
#define ARCANE_ITEMINDEXARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur un tableau d'index (localIds()) d'entités.
 *
 * \warning la vue n'est valide que tant que le tableau associé n'est
 * pas modifié. Les instances de cette classe sont en général temporaires
 * et ne doivent pas être conservées.
 *
 * En plus de la liste des entités, cette classe permet d'avoir des
 * informations supplémentaires comme par exemple si la liste est contigüe.
 */
class ARCANE_CORE_EXPORT ItemIndexArrayView
{
  friend class ItemVectorView;
  template <int Extent> friend class ItemConnectedListView;

 public:

  // NOTE: Si on ajoute des valeurs ici, il faut vérifier s'il faut les
  // propager dans les méthodes telles que subView().
  enum {
    F_Contigous = 1 << 1, //!< Les numéros locaux sont contigüs.
  };

 public:

  //! Construit une vue vide
  ItemIndexArrayView() : m_flags(0){}
  //! Construit une vue à partir des numéros locaux \a local_ids
  explicit ItemIndexArrayView(const Int32ConstArrayView local_ids)
  : m_local_ids(local_ids), m_flags(0) {}
  /*!
   * \brief Construit une vue à partir des numéros locaux \a local_ids avec
   * les informations \a flags.
   */
  ItemIndexArrayView(Int32ConstArrayView local_ids,Int32 aflags)
  : m_local_ids(local_ids), m_flags(aflags) {}

 public:

  //! Accède au \a i-ème élément du vecteur
  inline Int32 operator[](Integer index) const
  {
    return m_local_ids[index];
  }

  //! Nombre d'éléments du vecteur
  Int32 size() const { return m_local_ids.size(); }

  //! Ajoute à \a ids la liste des localIds() du vecteur.
  void fillLocalIds(Array<Int32>& ids) const;

  //! Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments
  inline ItemIndexArrayView subView(Integer abegin,Integer asize) const
  {
    // On propage le flag F_Contigous sur la sous-vue.
    // Pour les autres flags, il faudra vérifier s'il faut les propager.
    return ItemIndexArrayView(m_local_ids.subView(abegin,asize),m_flags);
  }

  Int32 flags() const { return m_flags; }

  //! Vrai si les localIds() sont contigüs
  bool isContigous() const { return m_flags & F_Contigous; }

 public:

  // TODO Rendre obsolète (3.11+)
  //! Tableau des numéros locaux des entités
  Int32ConstArrayView localIds() const { return m_local_ids; }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  operator Int32ConstArrayView() const
  {
    return m_local_ids;
  }

 private:

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* unguardedBasePointer() const
  {
    return m_local_ids.data();
  }

  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane. Do not use it")
  const Int32* data() const
  {
    return m_local_ids.data();
  }

 protected:
  
  Int32ConstArrayView m_local_ids;
  Int32 m_flags;

 private:

  const Int32* _data() const
  {
    return m_local_ids.data();
  }
  Int32ConstArrayView _localIds() const { return m_local_ids; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
