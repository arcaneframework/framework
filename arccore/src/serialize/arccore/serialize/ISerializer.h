// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializer.h                                               (C) 2000-2024 */
/*                                                                           */
/* Interface d'un serialiseur.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_ISERIALIZER_H
#define ARCCORE_SERIALIZE_ISERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializeGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/base/RefDeclarations.h"
#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un sérialiseur.
 *
 * Il est possible de créér une instance de cette classe via la méthode
 * createSerializer();
 *
 * Cette interface gère un sérialiseur pour stocker et relire un
 * ensemble de valeurs. La sérialisation se déroule en trois phases:
 *
 * 1. chaque objet appelle une ou plusieurs des méthodes reserve()/reserveSpan() pour
 *    indiquer de combien de mémoire il a besoin</li>
 * 2. la mémoire est allouée par allocateBuffer()</li>
 * 3. chaque objet appelle une ou plusieurs des méthodes put()/putSpan() pour ajouter
 *    au sérialiseur ses informations</li>
 *
 * La désérialisation se fait de manière identique mais utilise les fonctions
 * get()/getSpan(). Le fonctionnement est similaire à une file: à chaque
 * get()/getSpan() doit correspondre un précédent put()/putSpan() et les
 * get()/getSpan() et le put()/putSpan() doivent être dans le même ordre.
 *
 * Il est possible d'utiliser des surcharges de reserve()/get()/put(). Dans ce cas,
 * il faut assurer la cohérence de leur utilisation. Par exemple, si on fait
 * un reserveSpan(), il faut ensuite faire un putSpan() et un getSpan().
 *
 * \todo ajouter exemple.
 */
class ARCCORE_SERIALIZE_EXPORT ISerializer
{
 public:

  //! Mode de fonctionnement du sérialiseur
  enum eMode
  {
    ModeReserve, //! Le sérialiseur attend des reserve()
    ModePut, //!< Le sérialiseur attend des put()
    ModeGet //!< Le sérialiseur attend des get()
  };
  //! Mode de lecture du sérialiseur
  enum eReadMode
  {
    ReadReplace, //!< Replace les éléments actuels par ceux lus
    ReadAdd //!< Ajoute aux éléments actuels ceux lus
  };

  // NOTE: ne pas changer ces valeurs car elles sont utilisées dans Arcane
  enum eDataType
  {
    DT_Byte = 0, //!< Donnée de type octet
    DT_Real = 1, //!< Donnée de type réel
    DT_Int16 = 2, //!< Donnée de type entier 16 bits
    DT_Int32 = 3, //!< Donnée de type entier 32 bits
    DT_Int64 = 4, //!< Donnée de type entier 64 bits
    DT_Float32 = 12, //!< Donnée de type flottant 32 bits
    DT_Float16 = 11, //!< Donnée de type flottant 16 bits
    DT_BFloat16 = 10, //!< Donnée de type 'brain float'
    DT_Int8 = 13, //!< Donnée de type entier 8 bits
  };

  virtual ~ISerializer(){} //!< Libère les ressources

 public:

  /*!
   * \brief Réserve de la mémoire pour \a n valeurs de \a dt.
   *
   * \dt doit être un type intégral: DT_Int16, DT_Int32, DT_Int64,
   * DT_Real ou DT_Byte.
   *
   * Il faudra effectuer un appel à une méthode putSpan()
   * pour que la sérialisation soit correcte.
   */
  virtual void reserveSpan(eDataType dt, Int64 n) =0;

  //! \sa reserve(eDataType dt,Int64 n)
  void reserveSpan(int dt,Int64 n) { reserveSpan(static_cast<eDataType>(dt),n); }

  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Real> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Int16> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Int32> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Int64> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Byte> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Int8> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Float16> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const BFloat16> values);
  //! Réserve pour une vue de \a values éléments
  virtual void reserveSpan(Span<const Float32> values);

  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Real> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Int16> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Int32> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Int64> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Byte> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Int8> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Float16> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const Float32> values) =0;
  //! Réserve pour sauver le nombre d'éléments et les \a values éléments
  virtual void reserveArray(Span<const BFloat16> values) =0;

  /*!
   * \brief Réserve de la mémoire pour \a n objets de type \a dt.
   *
   * \dt doit être un type intégral: DT_Int16, DT_Int32, DT_Int64,
   * DT_Real ou DT_Byte.
   *
   * Il faudra effectuer \a n appels à une méthode put() avec une
   * seule valeur pour que la sérialisation soit correcte.
   *
   * Si on souhaite sérialiser plusieurs valeurs avec un seul
   * appel à put(), il faut utiliser la méthode reserveSpan().
   */
  virtual void reserve(eDataType dt,Int64 n) =0;

  //! \sa reserve(eDataType dt,Int64 n)
  void reserve(int dt,Int64 n) { reserve((eDataType)dt,n); }

  virtual void reserveInteger(Int64 n) =0;

  //! Réserve de la mémoire pour une chaîne de caractère \a str.
  virtual void reserve(const String& str) =0;

 public:
  //! Ajoute le tableau \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Real> values) =0;
  //! Ajoute le tableau \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int16> values) =0;
  //! Ajoute le tableau \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int32> values) =0;
  //! Ajoute le tableau \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Int64> values) =0;
  //! Ajoute le tableau \a values
  ARCCORE_DEPRECATED_2020("Use putSpan() instead")
  virtual void put(Span<const Byte> values) =0;

 public:
  //! Ajoute la chaîne \a value
  virtual void put(const String& value) =0;
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Real> values);
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Int16> values);
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Int32> values);
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Int64> values);
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Byte> values);
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Int8> values) =0;
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Float16> values) =0;
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const BFloat16> values) =0;
  //! Ajoute le tableau \a values
  virtual void putSpan(Span<const Float32> values) =0;

  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Real> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Int16> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Int32> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Int64> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Byte> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Int8> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Float16> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const BFloat16> values) =0;
  //! Sauve le nombre d'éléments et les \a values éléments
  virtual void putArray(Span<const Float32> values) =0;

  //! Ajoute \a value
  virtual void put(Real value) =0;
  //! Ajoute \a value
  virtual void put(Int16 value) =0;
  //! Ajoute \a value
  virtual void put(Int32 value) =0;
  //! Ajoute \a value
  virtual void put(Int64 value) =0;
  //! Ajoute value
  virtual void put(Byte value) =0;
  //! Ajoute value
  virtual void put(Int8 value) =0;
  //! Ajoute value
  virtual void put(Float16 value) =0;
  //! Ajoute value
  virtual void put(BFloat16 value) =0;
  //! Ajoute value
  virtual void put(Float32 value) =0;

  //! Ajoute le réel \a value
  virtual void putReal(Real value) =0;
  //! Ajoute l'entier \a value
  virtual void putInt16(Int16 value) =0;
  //! Ajoute l'entier \a value
  virtual void putInt32(Int32 value) =0;
  //! Ajoute l'entier \a value
  virtual void putInt64(Int64 value) =0;
  //! Ajoute l'entier \a value
  virtual void putInteger(Integer value) =0;
  //! Ajoute l'octet \a value
  virtual void putByte(Byte value) =0;
  //! Ajoute \a value
  virtual void putInt8(Int8 value) =0;
  //! Ajoute \a value
  virtual void putFloat16(Float16 value) =0;
  //! Ajoute \a value
  virtual void putBFloat16(BFloat16 value) =0;
  //! Ajoute \a value
  virtual void putFloat32(Float32 value) =0;

 public:
  //! Récupère le tableau \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(RealArrayView values) =0;
  //! Récupère le tableau \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(Int16ArrayView values) =0;
  //! Récupère le tableau \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(Int32ArrayView values) =0;
  //! Récupère le tableau \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(Int64ArrayView values) =0;
  //! Récupère le tableau \a values
  ARCCORE_DEPRECATED_2020("Use getSpan() instead")
  virtual void get(ByteArrayView values) =0;

 public:

  //! Récupère la chaîne \a value
  virtual void get(String& value) =0;
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Real> values);
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Int16> values);
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Int32> values);
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Int64> values);
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Byte> values);
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Int8> values) =0;
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Float16> values) =0;
  //! Récupère le tableau \a values
  virtual void getSpan(Span<BFloat16> values) =0;
  //! Récupère le tableau \a values
  virtual void getSpan(Span<Float32> values) =0;

  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Real>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Int16>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Int32>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Int64>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Byte>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Int8>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Float16>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<BFloat16>& values) =0;
  //! Redimensionne et remplit \a values
  virtual void getArray(Array<Float32>& values) =0;

  //! Récupère un réel
  virtual Real getReal() =0;
  //! Récupère un entier sur 16 bits
  virtual Int16 getInt16() =0;
  //! Récupère un entier naturel
  virtual Int32 getInt32() =0;
  //! Récupère une taille
  virtual Int64 getInt64() =0;
  //! Récupère une taille
  virtual Integer getInteger() =0;
  //! Récupère un octet
  virtual Byte getByte() =0;
  //! Récupère un Int8
  virtual Int8 getInt8() =0;
  //! Récupère un Float16
  virtual Float16 getFloat16() =0;
  //! Récupère un BFloat16
  virtual BFloat16 getBFloat16() =0;
  //! Récupère un Float32
  virtual Float32 getFloat32() =0;

  //! Alloue la mémoire du sérialiseur
  virtual void allocateBuffer() =0;

  ARCCORE_DEPRECATED_2020("Internal method. Do not use")
  virtual void allocateBuffer(Int64 nb_real,Int64 nb_int16,Int64 nb_int32,
                              Int64 nb_int64,Int64 nb_byte) =0;

  //! Mode de fonctionnement actuel
  virtual eMode mode() const =0;
  //! Positionne le fonctionnement actuel
  virtual void setMode(eMode new_mode) =0;

  //! Mode de lecture
  virtual eReadMode readMode() const =0;
  //! Positionne le mode de lecture
  virtual void setReadMode(eReadMode read_mode) =0;

  //! Copie les données de \a from dans cette instance
  virtual void copy(const ISerializer* from) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé une instance de ISerializer
 */
extern "C++" ARCCORE_SERIALIZE_EXPORT Ref<ISerializer>
createSerializer();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

