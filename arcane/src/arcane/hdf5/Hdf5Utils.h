// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5Utils.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires pour hdf5.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HDF5_HDF5UTILS_H
#define ARCANE_HDF5_HDF5UTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/datatype/DataTypes.h"

#include "arcane/hdf5/ArcaneHdf5Global.h"

// Cette macro pour MSVC avec les dll, pour eviter des symbols externes
// indéfinis avec H5T_NATIVE*
#define _HDF5USEDLL_
#include <hdf5.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il faut au moins hdf5 1.8
#if (H5_VERS_MAJOR<2) && (H5_VERS_MAJOR==1 && H5_VERS_MINOR<10)
#error "This version of HDF5 is too old. Version 1.10+ is required"
#endif

// Garde ces macros pour compatibilité mais il faudra les supprimer.
#define ARCANE_HDF5_1_6_AND_AFTER
#define ARCANE_HDF5_1_8_AND_AFTER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires pour Hdf5.
 */
namespace Hdf5Utils
{
extern "C"
{
  ARCANE_HDF5_EXPORT herr_t _ArcaneHdf5UtilsGroupIterateMe(hid_t,const char*,void*);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe servant d'initialiseur pour HDF.
 *
 * Cet objet permet d'initialiser de manière sure HDF5 en mode multi-thread.
 */
class ARCANE_HDF5_EXPORT HInit
{
 public:

  HInit();

 public:

  //! Vrai HDF5 est compilé avec le support de MPI
  static bool hasParallelHdf5();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t.
 *
 * Cette classe n'est pas copiable.
 */
class ARCANE_HDF5_EXPORT Hid
{
 public:

  Hid() = default;
  Hid(hid_t id)
  : m_id(id)
  {}
  virtual ~Hid() {}

 protected:

  // Il faudra interdire ce constructeur de recopie à terme
  Hid(const Hid& hid)
  : m_id(hid.id())
  {}
  void _setId(hid_t id) { m_id = id; }
  void _setNullId() { m_id = -1; }

 private:

  Hid& operator=(const Hid& hid) = delete;

 public:

  hid_t id() const { return m_id; }
  bool isBad() const { return m_id < 0; }

 private:

  hid_t m_id = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour une propriété (H5P*).
 */
class ARCANE_HDF5_EXPORT HProperty
: public Hid
{
 public:

  HProperty() { _setId(H5P_DEFAULT); }
  ~HProperty()
  {
    close();
  }
  HProperty(HProperty&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HProperty& operator=(HProperty&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }

 public:

  HProperty(const HProperty& v) = delete;
  HProperty& operator=(const HProperty& hid) = delete;

 public:

  void close()
  {
    if (id() > 0) {
      H5Pclose(id());
      _setNullId();
    }
  }

  void create(hid_t cls_id);
  void setId(hid_t new_id)
  {
    _setId(new_id);
  }

  /*!
   * \brief Créé une propriété de fichier pour MPIIO.
   *
   * Ne fonctionne que si HDF5 est compilé avec MPI. Sinon lance
   * une exception. Si \a mpi_comm est le communicateur MPI associé
   * à \a pm, l'appel à cette méthode créé une propriété comme suit:
   *
   * \code
   * create(H5P_FILE_ACCESS);
   * H5Pset_fapl_mpio(id(), mpi_comm, MPI_INFO_NULL);
   * \endcode
   */
  void createFilePropertyMPIIO(IParallelMng* pm);

  /*!
   * \brief Créé une propriété de dataset pour MPIIO.
   *
   * Ne fonctionne que si HDF5 est compilé avec MPI. Sinon lance
   * une exception. L'appel à cette méthode créé une propriété comme suit:
   *
   * \code
   * create(H5P_DATASET_XFER);
   * H5Pset_dxpl_mpio(id(), H5FD_MPIO_COLLECTIVE);
   * \endcode
   */
  void createDatasetTransfertCollectiveMPIIO();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un fichier.
 */
class ARCANE_HDF5_EXPORT HFile
: public Hid
{
 public:

  HFile() = default;
  ~HFile() { _close(); }
  HFile(HFile&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HFile& operator=(HFile&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HFile& operator=(const HFile& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HFile(const HFile& rhs)
  : Hid(rhs)
  {}

 public:

  void openTruncate(const String& var);
  void openAppend(const String& var);
  void openRead(const String& var);
  void openTruncate(const String& var, hid_t plist_id);
  void openAppend(const String& var, hid_t plist_id);
  void openRead(const String& var, hid_t plist_id);
  void close();

 private:

  herr_t _close();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe d'aide pour rechercher un groupe.
 */
class ARCANE_HDF5_EXPORT HGroupSearch
{
 public:
  HGroupSearch(const String& group_name)
  : m_group_name(group_name)
  {
  }
 public:
  herr_t iterateMe(const char* member_name)
  {
    //cerr << "** ITERATE <" << member_name << ">\n";
    if (m_group_name==member_name)
      return 1;
    return 0;
  }
 private:
  String m_group_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un groupe.
 */
class ARCANE_HDF5_EXPORT HGroup
: public Hid
{
 public:

  HGroup() {}
  ~HGroup() { close(); }
  HGroup(HGroup&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HGroup& operator=(HGroup&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HGroup& operator=(const HGroup& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HGroup(const HGroup& rhs)
  : Hid(rhs)
  {}

 public:

  void create(const Hid& loc_id, const String& group_name);
  void openOrCreate(const Hid& loc_id, const String& group_name);
  void recursiveCreate(const Hid& loc_id, const String& var);
  void recursiveCreate(const Hid& loc_id, const Array<String>& paths);
  void checkDelete(const Hid& loc_id, const String& var);
  void recursiveOpen(const Hid& loc_id, const String& var);
  void open(const Hid& loc_id, const String& var);
  void openIfExists(const Hid& loc_id, const Array<String>& var);
  bool hasChildren(const String& var);
  void close();
  static bool hasChildren(hid_t loc_id, const String& var);

 private:

  hid_t _checkOrCreate(hid_t loc_id, const String& group_name);
  hid_t _checkExist(hid_t loc_id, const String& group_name);

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un dataspace.
 */
class ARCANE_HDF5_EXPORT HSpace
: public Hid
{
 public:

  HSpace() {}
  explicit HSpace(hid_t id)
  : Hid(id)
  {}
  HSpace(HSpace&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  ~HSpace()
  {
    if (id() > 0)
      H5Sclose(id());
  }
  HSpace& operator=(HSpace&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HSpace& operator=(const HSpace& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HSpace(const HSpace& v)
  : Hid(v)
  {}

 public:

  void createSimple(int nb, hsize_t dims[]);
  void createSimple(int nb, hsize_t dims[], hsize_t max_dims[]);
  int nbDimension();
  herr_t getDimensions(hsize_t dims[], hsize_t max_dims[]);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un dataset.
 */
class ARCANE_HDF5_EXPORT HDataset
: public Hid
{
 public:

  HDataset() {}
  ~HDataset() { close(); }
  HDataset(HDataset&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HDataset& operator=(HDataset&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HDataset& operator=(const HDataset& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HDataset(const HDataset& v)
  : Hid(v)
  {}

 public:

  void close()
  {
    if (id() > 0)
      H5Dclose(id());
    _setNullId();
  }
  void create(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id, hid_t plist);
  void create(const Hid& loc_id,const String& var,hid_t save_type,
              const HSpace& space_id,const HProperty& link_plist,
              const HProperty& creation_plist,const HProperty& access_plist);
  void recursiveCreate(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id, hid_t plist);
  void open(const Hid& loc_id, const String& var);
  void openIfExists(const Hid& loc_id, const String& var);
  herr_t write(hid_t native_type, const void* array);
  herr_t write(hid_t native_type, const void* array, const HSpace& memspace_id,
               const HSpace& filespace_id, hid_t plist);
  herr_t write(hid_t native_type, const void* array, const HSpace& memspace_id,
               const HSpace& filespace_id, const HProperty& plist);
  herr_t read(hid_t native_type, void* array)
  {
    return H5Dread(id(), native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
  }
  void readWithException(hid_t native_type, void* array);
  HSpace getSpace();
  herr_t setExtent(const hsize_t new_dims[]);

 private:

  void _remove(hid_t hid, const String& var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un attribute.
 */
class ARCANE_HDF5_EXPORT HAttribute
: public Hid
{
 public:

  HAttribute() {}
  ~HAttribute()
  {
    if (id() > 0)
      H5Aclose(id());
  }
  HAttribute(HAttribute&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HAttribute& operator=(HAttribute&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HAttribute& operator=(const HAttribute& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HAttribute(const HAttribute& v)
  : Hid(v)
  {}

 public:

  void remove(const Hid& loc_id, const String& var)
  {
    _setId(H5Adelete(loc_id.id(), var.localstr()));
  }
  void create(const Hid& loc_id, const String& var, hid_t save_type, const HSpace& space_id)
  {
    _setId(H5Acreate2(loc_id.id(), var.localstr(), save_type, space_id.id(), H5P_DEFAULT, H5P_DEFAULT));
  }
  void open(const Hid& loc_id, const String& var)
  {
    _setId(H5Aopen_name(loc_id.id(), var.localstr()));
  }
  herr_t write(hid_t native_type, void* array)
  {
    return H5Awrite(id(), native_type, array);
  }
  herr_t read(hid_t native_type, void* array)
  {
    return H5Aread(id(), native_type, array);
  }
  HSpace getSpace()
  {
    return HSpace(H5Aget_space(id()));
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un hid_t pour un type.
 */
class ARCANE_HDF5_EXPORT HType
: public Hid
{
 public:

  HType() {}
  ~HType()
  {
    if (id() > 0)
      H5Tclose(id());
  }
  HType(HType&& rhs)
  : Hid(rhs.id())
  {
    rhs._setNullId();
  }
  HType& operator=(HType&& rhs)
  {
    _setId(rhs.id());
    rhs._setNullId();
    return (*this);
  }
  HType& operator=(const HType& hid) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  HType(const HType& v)
  : Hid(v)
  {}

 public:

  void setId(hid_t new_id)
  {
    _setId(new_id);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Définition des types standards Arcane pour hdf5.
 *
 * Une instance de cette classe construit des types HDF5 pour faire la
 * conversion entre les types HDF5 et les types Arcane.
 *
 * Le constructeur par défaut utilisant des appels HDF5, il n'est pas thread-safe.
 * Si on est en contexte multi-thread, il est préférable d'utiliser
 * StandardTypes(false) et d'appeler init() pour initialiser les types.
 */
class ARCANE_HDF5_EXPORT StandardTypes
{
 public:

  /*!
   * \brief Créé une instance en initialisant les types.
   *
   * \warning non thread-safe.
   */
  StandardTypes();

  //! Créé une instance sans initialiser les types is \a do_init est faux.
  explicit StandardTypes(bool do_init);

  ARCANE_DEPRECATED_REASON("Y2023: Copy constructor is deprecated. This class has unique ownership")
  StandardTypes(const StandardTypes& rhs) = default;

  ~StandardTypes();

  StandardTypes& operator=(const StandardTypes& rhs) = delete;

 public:

  //! Initialise les types.
  void initialize();

 public:

  hid_t nativeType(float) const { return H5T_NATIVE_FLOAT; }
  hid_t nativeType(double) const { return H5T_NATIVE_DOUBLE; }
  hid_t nativeType(Real2) const { return m_real2_id.id(); }
  hid_t nativeType(Real3) const { return m_real3_id.id(); }
  hid_t nativeType(Real2x2) const { return m_real2x2_id.id(); }
  hid_t nativeType(Real3x3) const { return m_real3x3_id.id(); }
  hid_t nativeType(long double) const { return H5T_NATIVE_LDOUBLE; }
  hid_t nativeType(unsigned int) const { return H5T_NATIVE_UINT; }
  hid_t nativeType(unsigned long) const { return H5T_NATIVE_ULONG; }
  hid_t nativeType(unsigned long long) const { return H5T_NATIVE_ULLONG; }
  hid_t nativeType(int) const { return H5T_NATIVE_INT; }
  hid_t nativeType(long long) const { return H5T_NATIVE_LLONG; }
  hid_t nativeType(long) const { return H5T_NATIVE_LONG; }
  hid_t nativeType(char) const { return H5T_NATIVE_CHAR; }
  hid_t nativeType(unsigned char) const { return H5T_NATIVE_UCHAR; }
  hid_t nativeType(signed char) const { return H5T_NATIVE_SCHAR; }
  hid_t nativeType(unsigned short) const { return H5T_NATIVE_USHORT; }
  hid_t nativeType(short) const { return H5T_NATIVE_SHORT; }
#ifdef ARCANE_REAL_NOT_BUILTIN
  hid_t nativeType(Real) const;
#endif
  hid_t nativeType(eDataType sd) const;

 public:

  hid_t saveType(float) const
  {
    return m_real_id.id();
  }
  hid_t saveType(double) const
  {
    return m_real_id.id();
  }
  hid_t saveType(Real2) const
  {
    return m_real2_id.id();
  }
  hid_t saveType(Real3) const
  {
    return m_real3_id.id();
  }
  hid_t saveType(Real2x2) const
  {
    return m_real2x2_id.id();
  }
  hid_t saveType(Real3x3) const
  {
    return m_real3x3_id.id();
  }
  hid_t saveType(long double) const
  {
    return m_real_id.id();
  }
  hid_t saveType(short) const
  {
    return m_short_id.id();
  }
  hid_t saveType(unsigned short) const
  {
    return m_ushort_id.id();
  }
  hid_t saveType(unsigned int) const
  {
    return m_uint_id.id();
  }
  hid_t saveType(unsigned long) const
  {
    return m_ulong_id.id();
  }
  hid_t saveType(unsigned long long) const
  {
    return m_ulong_id.id();
  }
  hid_t saveType(int) const
  {
    return m_int_id.id();
  }
  hid_t saveType(long) const
  {
    return m_long_id.id();
  }
  hid_t saveType(long long) const
  {
    return m_long_id.id();
  }
  hid_t saveType(char) const
  {
    return m_char_id.id();
  }
  hid_t saveType(unsigned char) const
  {
    return m_uchar_id.id();
  }
  hid_t saveType(signed char) const
  {
    return m_char_id.id();
  }
#ifdef ARCANE_REAL_NOT_BUILTIN
  hid_t saveType(Real) const
  {
    return m_real_id.id();
  }
#endif
  hid_t saveType(eDataType sd) const;

 private:

  /*!
   * \brief Classe initialisant HDF.
   *
   * \warning Cette instance doit toujours être définie avant les membres qui
   * utilisent HDF5 pour que l'initialisation est lieu en premier et la libération
   * des ressources en dernier.
   */
  HInit m_init;

 public:

  HType m_char_id; //!< Identifiant HDF des entiers signés
  HType m_uchar_id; //!< Identifiant HDF des caractères non-signés
  HType m_short_id; //!< Identifiant HDF des entiers signés
  HType m_ushort_id; //!< Identifiant HDF des entiers long signés
  HType m_int_id; //!< Identifiant HDF des entiers signés
  HType m_long_id; //!< Identifiant HDF des entiers long signés
  HType m_uint_id; //!< Identifiant HDF des entiers non signés
  HType m_ulong_id; //!< Identifiant HDF des entiers long non signés
  HType m_real_id; //!< Identifiant HDF des réels
  HType m_real2_id; //!< Identifiant HDF pour les Real2
  HType m_real3_id; //!< Identifiant HDF pour les Real3
  HType m_real2x2_id; //!< Identifiant HDF pour les Real2x2
  HType m_real3x3_id; //!< Identifiant HDF pour les Real3x3

 private:

  void _H5Tinsert(hid_t type, const char* name, Integer offset, hid_t field_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un dataset simple d'un fichier HDF5 qui représente
 * un tableau.
 */
class ARCANE_HDF5_EXPORT StandardArray
{
 public:

  StandardArray(hid_t hfile, const String& hpath);
  virtual ~StandardArray() {}

 public:

  /*!
   * \brief En lecture, positionne le chemin dans \a hfile du dataset contenant les unique_ids.
   *
   * Cet appel est optionnel mais s'il est utilisé, il doit l'être avant
   * de lire les valeurs.
   */
  void setIdsPath(const String& ids_path);
  void readDim();
  Int64ConstArrayView dimensions() const { return m_dimensions; }
  virtual bool exists() const;

 protected:

  void _write(const void* buffer, Integer nb_element, hid_t save_type, hid_t native_type);

 protected:

  hid_t m_hfile;
  String m_hpath;
  String m_ids_hpath;
  HDataset m_hdataset;
  HDataset m_ids_dataset;
  Int64UniqueArray m_dimensions;
  bool m_is_init;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Encapsule un dataset simple d'un fichier HDF5 qui représente
 * un tableau.
 */
template<typename DataType>
class ARCANE_HDF5_EXPORT StandardArrayT
: public StandardArray
{
 private:
  struct ValueWithUid
  {
   public:
    Int64 m_uid;
    Integer m_index;
   public:
    bool operator<(const ValueWithUid& rhs) const
    {
      return m_uid < rhs.m_uid;
    }
  };
 public:
  StandardArrayT(hid_t hfile,const String& hpath);
 public:
  /*!
   * \brief Lit le dataset d'un tableau 1D.
   * Cette opération n'est valide qu'après un appel à readDim().
   * \a buffer doit avoir été alloué.
   * Pour lire directement, utiliser directRead()
   */
  void read(StandardTypes& st,ArrayView<DataType> buffer);
  /*!
   * \brief Lit le dataset d'un tableau 1D.
   */
  void directRead(StandardTypes& st,Array<DataType>& buffer);
  void parallelRead(IParallelMng* pm,StandardTypes& st,
                    Array<DataType>& buffer,Int64Array& unique_ids);
  void write(StandardTypes& st,ConstArrayView<DataType> buffer);
  void parallelWrite(IParallelMng* pm,StandardTypes& st,
                     ConstArrayView<DataType> buffer,
                     Int64ConstArrayView unique_ids);
 private:
  void _writeSortedValues(ITraceMng* tm,StandardTypes& st,ConstArrayView<DataType> buffer,
                          Int64ConstArrayView unique_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Encapsule un dataset simple d'un fichier HDF5 qui représente un scalaire (éventuellement String).
 */
template<typename DataType>
class ARCANE_HDF5_EXPORT StandardScalarT
{
 public:
  //! Constructeur
  StandardScalarT(hid_t hfile,const String& hpath) : m_hfile(hfile), m_hpath(hpath) { }
 public:
  //! Lit une donnée
  DataType read(Hdf5Utils::StandardTypes& st);

  //! Ecrit une donnée
  void write(Hdf5Utils::StandardTypes& st, const DataType & t);

 protected:
  hid_t m_hfile;
  String m_hpath;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
