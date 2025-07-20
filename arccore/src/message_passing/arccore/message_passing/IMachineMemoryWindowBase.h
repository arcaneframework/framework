// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMachineMemoryWindowBase.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface de classe permettant de créer une fenêtre mémoire pour un noeud */
/* de calcul. Cette fenêtre sera contigüe en mémoire.                        */
/*---------------------------------------------------------------------------*/

#ifndef ARCCORE_MESSAGEPASSING_IMACHINEMEMORYWINDOWBASE_H
#define ARCCORE_MESSAGEPASSING_IMACHINEMEMORYWINDOWBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de créer une fenêtre mémoire pour un noeud
 * de calcul.
 *
 * Cette fenêtre sera contigüe en mémoire et sera accessible par
 * tous les processus du noeud.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMachineMemoryWindowBase
{
 public:

  virtual ~IMachineMemoryWindowBase() = default;

 public:

  /*!
   * \brief Méthode permettant d'obtenir la taille d'un élement de la fenêtre.
   *
   * \return La taille d'un élement.
   */
  virtual Integer sizeofOneElem() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille de son segment de la
   * fenêtre mémoire utilisable (en nombre d'éléments).
   *
   * \return La taille du segment.
   */
  virtual Integer sizeSegment() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille du segment d'un processus
   * du noeud de la fenêtre mémoire utilisable (en nombre
   * d'éléments).
   *
   * \return La taille du segment.
   */
  virtual Integer sizeSegment(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille de la fenêtre mémoire
   * utilisable (en nombre d'éléments).
   *
   * \return La taille de la fenêtre.
   */
  virtual Integer sizeWindow() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir un pointeur vers son segment de la
   * fenêtre mémoire.
   *
   * \return Un pointeur (ne pas détruire).
   */
  virtual void* dataSegment() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir un pointeur vers le segment d'un
   * processus du noeud de la fenêtre mémoire.
   *
   * \return Un pointeur (ne pas détruire).
   */
  virtual void* dataSegment(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir un pointeur vers la fenêtre mémoire.
   *
   * \return Un pointeur (ne pas détruire).
   */
  virtual void* dataWindow() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille et un pointeur de son
   * segment (la taille en nombre d'éléments).
   *
   * \return Une paire [taille, ptr].
   */
  virtual std::pair<Integer, void*> sizeAndDataSegment() const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille et un pointeur du segment
   * d'un processus du noeud (la taille en nombre d'éléments).
   *
   * \return Une paire [taille, ptr].
   */
  virtual std::pair<Integer, void*> sizeAndDataSegment(Int32 rank) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la taille et un pointeur de la
   * fenêtre (la taille en nombre d'éléments).
   *
   * \return Une paire [taille, ptr].
   */
  virtual std::pair<Integer, void*> sizeAndDataWindow() const = 0;

  /*!
   * \brief Méthode permettant de redimensionner les segments de la fenêtre.
   * Appel collectif.
   *
   * La taille totale de la fenêtre doit être inférieure ou égale à la taille
   * d'origine.
   *
   * \param new_nb_elem La nouvelle taille de notre segment.
   */
  virtual void resizeSegment(Integer new_nb_elem) = 0;

  /*!
   * \brief Méthode permettant d'obtenir les rangs qui possèdent un segment
   * dans la fenêtre.
   *
   * L'ordre des processus de la vue retournée correspond à l'ordre des
   * segments dans la fenêtre.
   *
   * \return Une vue contenant les ids des rangs.
   */
  virtual ConstArrayView<Int32> machineRanks() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

