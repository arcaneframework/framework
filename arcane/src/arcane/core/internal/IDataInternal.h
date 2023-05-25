// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataInternal.h                                             (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de IData.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
#define ARCANE_CORE_INTERNAL_IDATAINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataCompressor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DataCompressionBuffer
{
 public:

  UniqueArray<std::byte> m_buffer;
  Int64 m_original_dim1_size = 0;
  Int64 m_original_dim2_size = 0;
  IDataCompressor* m_compressor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IData.
 */
class ARCANE_CORE_EXPORT IDataInternal
{
 public:

  virtual ~IDataInternal() = default;

 public:

  /*!
   * \brief Compresse les données et libère la mémoire associée
   *
   * Compresse les données et remplit \a buf avec les information compressées.
   * Libère ensuite la mémoire associée. L'instance ne sera plus utilisable
   * tant que decompressAndFill() n'aura pas été appelé.
   *
   * \retval true si une compression a eu lieu.
   * \retval false si l'instance ne supporte pas la compression. Dans ce cas
   * elle reste utilisable.
   *
   * \warning L'appel à cette méthode modifie le conteneur sous-jacent. Si
   * cette donnée est associée à une variable il faut appeler IVariable::syncReferences().
   */
  virtual bool compressAndClear(DataCompressionBuffer& buf)
  {
    ARCANE_UNUSED(buf);
    return false;
  }

  /*!
   * \brief Décompresse les données et remplit les valeurs de la donnée.
   *
   * Décompresse les données de \a buf et remplit les valeurs de cette instance
   * avec les information decompressées.
   *
   * \retval true si une décompression a eu lieu.
   * \retval false si aucune décompression n'a eu lieu car l'instance ne le
   * supporte pas.
   *
   * \warning L'appel à cette méthode modifie le conteneur sous-jacent. Si
   * cette donnée est associée à une variable il faut appeler IVariable::syncReferences().
   */
  virtual bool decompressAndFill(DataCompressionBuffer& buf)
  {
    ARCANE_UNUSED(buf);
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullDataInternal
: public IDataInternal
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau d'un type \a T
 */
template <class DataType>
class IArrayDataInternalT
: public IDataInternal
{
 public:

  //! Réserve de la mémoire pour \a new_capacity éléments
  virtual void reserve(Integer new_capacity) =0;

  //! Conteneur associé à la donnée.
  virtual Array<DataType>& _internalDeprecatedValue() = 0;

  //! Capacité allouée par le conteneur
  virtual Integer capacity() const =0;

  //! Libère la mémoire additionnelle éventuellement allouée
  virtual void shrink() const =0;

  //! Redimensionne le conteneur.
  virtual void resize(Integer new_size) =0;

  //! Vide le conteneur et libère la mémoire alloué.
  virtual void dispose() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau bi-dimensionnel d'un type \a T
 */
template <class DataType>
class IArray2DataInternalT
: public IDataInternal
{
 public:

  //! Réserve de la mémoire pour \a new_capacity éléments
  virtual void reserve(Integer new_capacity) = 0;

  //! Conteneur associé à la donnée.
  virtual Array2<DataType>& _internalDeprecatedValue() = 0;

  //! Redimensionne le conteneur.
  virtual void resizeOnlyDim1(Int32 new_dim1_size) = 0;

  //! Redimensionne le conteneur.
  virtual void resize(Int32 new_dim1_size, Int32 new_dim2_size) = 0;

  //! Libère la mémoire additionnelle éventuellement allouée
  virtual void shrink() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
