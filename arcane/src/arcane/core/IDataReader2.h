// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataReader2.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface de lecture des données d'une variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDATAREADER2_H
#define ARCANE_CORE_IDATAREADER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableMetaData;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de relecture des données.
 */
class ARCANE_CORE_EXPORT DataReaderInfo
{
 public:
  DataReaderInfo(){}
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de relecture des données d'une variable.
 */
class ARCANE_CORE_EXPORT VariableDataReadInfo
{
 public:
  VariableDataReadInfo(VariableMetaData* varmd,IData* data)
  : m_varmd(varmd), m_data(data){}
 public:
  VariableMetaData* variableMetaData() const { return m_varmd; }
  IData* data() const { return m_data; }
 private:
  VariableMetaData* m_varmd;
  IData* m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Interface de lecture des données d'une variable (Version 2)
 *
 * Cette interface permet de lire les données d'une variable à partir
 * d'un fichier de protection.
 *
 * Cette interface est en générale utilisée par
 * IVariableMng::readCheckpoint(). L'ordre d'appel des opérations
 * est le suivant:
 * \code
 * IDataReader2 reader = ...;
 * DataReaderInfo read_infos = ...
 * reader->fillMetaData(...);
 * reader->beginRead(read_infos);
 * for( const VariableDataReadInfo& i : variables )
 *   reader->read(i);
 * reader->endRead();
 * \endcode
 */
 class IDataReader2
{
 public:

  //! Libère les ressources
  virtual ~IDataReader2(){}

 public:

  //! Remplit \a bytes avec le contenu des méta-données
  virtual void fillMetaData(ByteArray& bytes) =0;
  //! Notifie du début de lecture des données
  virtual void beginRead(const DataReaderInfo& infos) =0;
  //! Lit les données des informations spécifiées par \a infos
  virtual void read(const VariableDataReadInfo& infos) =0;
  //! Notifie de la fin de lecture des données
  virtual void endRead() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

