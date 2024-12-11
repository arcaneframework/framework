// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5Utils.cc                                                (C) 2000-2024 */
/*                                                                           */
/* Utilitaires HDF5.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/IOException.h"

#include "arcane/ArcaneException.h"
#include "arcane/IParallelMng.h"

#include "arcane/hdf5/Hdf5Utils.h"

#include <algorithm>

#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Hdf5Utils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
std::once_flag h5open_once_flag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HInit::
HInit()
{
  // Garanti que cela ne sera appelé qu'une seule fois et protège des appels
  // concurrents.
  std::call_once(h5open_once_flag, [](){ H5open(); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool HInit::
hasParallelHdf5()
{
#ifdef H5_HAVE_PARALLEL
  return true;
#else
  return false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

hid_t _H5Gopen(hid_t loc_id, const char *name)
{
  return H5Gopen2(loc_id,name,H5P_DEFAULT);
}

hid_t _H5Gcreate(hid_t loc_id, const char *name)
{
  return H5Gcreate2(loc_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_HDF5_EXPORT herr_t
_ArcaneHdf5UtilsGroupIterateMe(hid_t g,const char* mn,void* ptr)
{
  ARCANE_UNUSED(g);
  HGroupSearch* rw = reinterpret_cast<HGroupSearch*>(ptr);
  return rw->iterateMe(mn);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
splitString(const String& str,Array<String>& str_array,char c)
{
  const char* str_str = str.localstr();
  Int64 offset = 0;
  Int64 len = str.length();
  for( Int64 i=0; i<len; ++i ){
    if (str_str[i]==c && i!=offset){
      str_array.add(std::string_view(str_str+offset,i-offset));
      offset = i+1;
    }
  }
  if (len!=offset)
    str_array.add(std::string_view(str_str+offset,len-offset));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HFile::
openTruncate(const String& var)
{
  close();
  _setId(H5Fcreate(var.localstr(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

void HFile::
openAppend(const String& var)
{
  close();
  _setId(H5Fopen(var.localstr(),H5F_ACC_RDWR,H5P_DEFAULT));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

void HFile::
openRead(const String& var)
{
  close();
  _setId(H5Fopen(var.localstr(),H5F_ACC_RDONLY,H5P_DEFAULT));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

void HFile::
openTruncate(const String& var,hid_t plist_id)
{
  close();
  _setId(H5Fcreate(var.localstr(),H5F_ACC_TRUNC,H5P_DEFAULT,plist_id));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

void HFile::
openAppend(const String& var,hid_t plist_id)
{
  close();
  _setId(H5Fopen(var.localstr(),H5F_ACC_RDWR,plist_id));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

void HFile::
openRead(const String& var,hid_t plist_id)
{
  close();
  _setId(H5Fopen(var.localstr(),H5F_ACC_RDONLY,plist_id));
  if (isBad())
    ARCANE_THROW(ReaderWriterException,"Can not open file '{0}'",var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HFile::
_close()
{
  herr_t e = 0;
  if (id()>0){
    e = H5Fclose(id());
    _setNullId();
  }
  return e;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HFile::
close()
{
  herr_t e = _close();
  if (e<0)
    ARCANE_THROW(ReaderWriterException,"Can not close file");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
recursiveCreate(const Hid& loc_id,const String& var)
{
  UniqueArray<String> bufs;
  splitString(var,bufs,'/');
  recursiveCreate(loc_id,bufs);
}

void HGroup::
recursiveCreate(const Hid& loc_id,const Array<String>& bufs)
{
  close();
  hid_t last_hid = loc_id.id();
  Integer nb_create = bufs.size();
  UniqueArray<hid_t> ref_ids(nb_create);
  for( Integer i=0; i<nb_create; ++i ){
    last_hid = _checkOrCreate(last_hid,bufs[i]);
    ref_ids[i] = last_hid;
  }
  // Libere tous les groupes intermediaires crees
  for( Integer i=0; i<nb_create-1; ++i )
    H5Gclose(ref_ids[i]);
  _setId(last_hid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
checkDelete(const Hid& loc_id,const String& var)
{
  UniqueArray<String> bufs;
  splitString(var,bufs,'/');
  hid_t last_hid = loc_id.id();
  hid_t parent_hid = last_hid;
  Integer i = 0;
  Integer size = bufs.size();
  for( ; i<size; ++i ){
    parent_hid = last_hid;
    last_hid = _checkExist(last_hid,bufs[i]);
    if (last_hid==0)
      break;
  }
  // Groupe trouvé, on le détruit.
  if (last_hid>0 && parent_hid>0 && i==size){
    //cerr << "** DELETE <" << bufs[size-1] << "\n";
    H5Gunlink(parent_hid,bufs[size-1].localstr());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
recursiveOpen(const Hid& loc_id,const String& var)
{
  close();
  UniqueArray<String> bufs;
  splitString(var,bufs,'/');
  hid_t last_hid = loc_id.id();
  Integer nb_open = bufs.size();
  UniqueArray<hid_t> ref_ids(nb_open);
  for( Integer i=0; i<nb_open; ++i ){
    last_hid = _H5Gopen(last_hid,bufs[i].localstr());
    ref_ids[i] = last_hid;
  }
  // Libere tous les groupes intermediaires ouverts
  for( Integer i=0; i<nb_open-1; ++i )
    H5Gclose(ref_ids[i]);
  _setId(last_hid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
openIfExists(const Hid& loc_id,const Array<String>& paths)
{
  close();
  hid_t last_hid = loc_id.id();
  bool is_valid = true;
  Integer nb_open = paths.size();
  UniqueArray<hid_t> ref_ids;
  ref_ids.reserve(nb_open);
  for( Integer i=0; i<nb_open; ++i ){
    if (HGroup::hasChildren(last_hid,paths[i].localstr())){
      last_hid = _H5Gopen(last_hid,paths[i].localstr());
      ref_ids.add(last_hid);
    }
    else{
      is_valid = false;
      break;
    }
  }
  if (is_valid)
    _setId(last_hid);
  // Ferme tous les groupes intermediaires
  for( Integer i=0; i<ref_ids.size(); ++i ){
    if (ref_ids[i]!=last_hid)
      H5Gclose(ref_ids[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool HGroup::
hasChildren(const String& var)
{
  return hasChildren(id(),var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool HGroup::
hasChildren(hid_t loc_id,const String& var)
{
  HGroupSearch gs(var);
  herr_t v = H5Giterate(loc_id,".",0,_ArcaneHdf5UtilsGroupIterateMe,&gs);
  bool has_children = v>0;
  //cout << "** HAS CHILDREN " << var << " v=" << has_children << '\n';
  return has_children;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

hid_t HGroup::
_checkOrCreate(hid_t loc_id,const String& group_name)
{
  // Pour vérifier si un groupe existe déjà, comme il n'existe aucune
  // fonction digne de ce nom dans HDF5, on utilise le mécanisme d'itération
  // pour stocker tous les groupes fils de ce groupe, et on recherche ensuite
  // si le groupe souhaité existe
  HGroupSearch gs(group_name);
  //cerr << "** CHECK CREATE <" << group_name.str()  << ">\n";
  herr_t v = H5Giterate(loc_id,".",0,_ArcaneHdf5UtilsGroupIterateMe,&gs);

  // Regarde si le groupe existe déjà
  //herr_t he = H5Gget_objinfo(loc_id,group_name.str(),true,0);
  //cerr << "** CHECK CREATE <" << group_name.str()  << "> " << v << "\n";
  //cerr << "** CHECK CREATE <" << group_name.str()  << "> " << v << ' ' << he << "\n";
  if (v>0){
    return _H5Gopen(loc_id,group_name.localstr());
  }
  hid_t new_id = _H5Gcreate(loc_id,group_name.localstr());
  //cerr << "** TRY TO CREATE <" << group_name.str()  << "> " << new_id << "\n";
  return new_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
create(const Hid& loc_id, const String& group_name)
{
  _setId(H5Gcreate2(loc_id.id(), group_name.localstr(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
openOrCreate(const Hid& loc_id, const String& group_name)
{
  hid_t id = _checkOrCreate(loc_id.id(),group_name);
  if (id<0)
    ARCANE_THROW(ReaderWriterException,"Can not open or create group named '{0}'",group_name);
  _setId(id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
open(const Hid& loc_id,const String& var)
{
  hid_t id = _H5Gopen(loc_id.id(),var.localstr());
  if (id<0)
    ARCANE_THROW(ReaderWriterException,"Can not find group named '{0}'",var);
  _setId(id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HGroup::
close()
{
  if (id()>0){
    H5Gclose(id());
    _setNullId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

hid_t HGroup::
_checkExist(hid_t loc_id,const String& group_name)
{
  // Pour vérifier si un groupe existe déjà, comme il n'existe aucune
  // fonction digne de ce nom dans HDF5, on utilise le mécanisme d'itération
  // pour stocker tous les groupes fils de ce groupe, et on recherche ensuite
  // si le groupe souhaité existe
  HGroupSearch gs(group_name);
  //cerr << "** CHECK CREATE <" << group_name.str()  << ">\n";
  herr_t v = H5Giterate(loc_id,".",0,_ArcaneHdf5UtilsGroupIterateMe,&gs);

  // Regarde si le groupe existe déjà
  //herr_t he = H5Gget_objinfo(loc_id,group_name.str(),true,0);
  //cerr << "** CHECK CREATE <" << group_name.str()  << "> " << v << "\n";
  //cerr << "** CHECK CREATE <" << group_name.str()  << "> " << v << ' ' << he << "\n";
  if (v>0){
    return _H5Gopen(loc_id,group_name.localstr());
  }
  //hid_t new_id = H5Gcreate(loc_id,group_name.localstr(),0);
  //cerr << "** TRY TO CREATE <" << group_name.str()  << "> " << new_id << "\n";
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HSpace::
createSimple(int nb, hsize_t dims[])
{
  _setId(H5Screate_simple(nb, dims, nullptr));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HSpace::
createSimple(int nb, hsize_t dims[], hsize_t max_dims[])
{
  _setId(H5Screate_simple(nb, dims, max_dims));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int HSpace::
nbDimension()
{
 return H5Sget_simple_extent_ndims(id());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HSpace::
getDimensions(hsize_t dims[], hsize_t max_dims[])
{
  return H5Sget_simple_extent_dims(id(), dims, max_dims);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
create(const Hid& loc_id,const String& var,hid_t save_type,
       const HSpace& space_id,hid_t plist)
{
	hid_t hid = H5Dcreate2(loc_id.id(),var.localstr(),save_type,space_id.id(),
                        plist,H5P_DEFAULT,H5P_DEFAULT);
  //cerr << "** CREATE ID=" << hid << '\n';
  _setId(hid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
create(const Hid& loc_id,const String& var,hid_t save_type,
       const HSpace& space_id,const HProperty& link_plist,
       const HProperty& creation_plist,const HProperty& access_plist)
{
	hid_t hid = H5Dcreate2(loc_id.id(),var.localstr(),save_type,space_id.id(),
                         link_plist.id(),creation_plist.id(),access_plist.id());
  //cerr << "** CREATE ID=" << hid << '\n';
  _setId(hid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HDataset::
write(hid_t native_type,const void* array)
{
  //cerr << "** WRITE ID=" << id() << '\n';
	return H5Dwrite(id(),native_type,H5S_ALL,H5S_ALL,H5P_DEFAULT,array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HDataset::
write(hid_t native_type,const void* array,const HSpace& memspace_id,
      const HSpace& filespace_id,hid_t plist)
{
  //cerr << "** WRITE ID=" << id() << '\n';
  return H5Dwrite(id(),native_type,memspace_id.id(),filespace_id.id(),plist,array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HDataset::
write(hid_t native_type,const void* array,const HSpace& memspace_id,
      const HSpace& filespace_id,const HProperty& plist)
{
  //cerr << "** WRITE ID=" << id() << '\n';
  return H5Dwrite(id(),native_type,memspace_id.id(),filespace_id.id(),plist.id(),array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
readWithException(hid_t native_type,void* array)
{
	herr_t err = H5Dread(id(),native_type,H5S_ALL,H5S_ALL,H5P_DEFAULT,array);
  if (err!=0)
    ARCANE_THROW(IOException,"Can not read dataset");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HSpace HDataset::
getSpace()
{
  return HSpace(H5Dget_space(id()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t HDataset::
setExtent(const hsize_t new_dims[])
{
  return H5Dset_extent(id(),new_dims);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
recursiveCreate(const Hid& loc_id,const String& var,hid_t save_type,
                const HSpace& space_id,hid_t plist)
{
  // Si le dataset existe déjà, il faut le supprimer
  // car sinon il n'est pas toujours possible de modifer le space_id
  UniqueArray<String> paths;
  splitString(var,paths,'/');
  Integer nb_path = paths.size();
  if (nb_path==1){
    if (HGroup::hasChildren(loc_id.id(),var)){
      _remove(loc_id.id(),var);
    }
    create(loc_id,var,save_type,space_id,plist);
    return;
  }
  String last_name = paths[nb_path-1];
  paths.resize(nb_path-1);
  HGroup group;
  group.recursiveCreate(loc_id,paths);
  if (group.hasChildren(last_name)){
    _remove(group.id(),last_name);
  }
  create(group.id(),last_name,save_type,space_id,plist);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
_remove(hid_t hid,const String& var)
{
  H5Gunlink(hid,var.localstr());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
open(const Hid& loc_id,const String& var)
{
	_setId(H5Dopen2(loc_id.id(),var.localstr(),H5P_DEFAULT));
  if(isBad())
    ARCANE_THROW(IOException,"Can not open dataset '{0}'",var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HDataset::
openIfExists(const Hid& loc_id,const String& var)
{
  UniqueArray<String> paths;
  splitString(var,paths,'/');
  Integer nb_path = paths.size();
  HGroup parent_group;
  String last_name = var;
  if (nb_path>1){
    last_name = paths[nb_path-1];
    paths.resize(nb_path-1);
    parent_group.openIfExists(loc_id,paths);
  }
  else{
    parent_group.open(loc_id,".");
  }
  if (parent_group.isBad())
    return;
  if (parent_group.hasChildren(last_name))
    open(loc_id.id(),var.localstr());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HProperty::
create(hid_t cls_id)
{
  close();
  _setId(H5Pcreate(cls_id));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HProperty::
createFilePropertyMPIIO(IParallelMng* pm)
{
#ifdef H5_HAVE_PARALLEL
  void* arcane_comm = pm->getMPICommunicator();
  if (!arcane_comm)
    ARCANE_FATAL("No MPI environment available");
  MPI_Comm mpi_comm = *((MPI_Comm*)arcane_comm);
  MPI_Info mpi_info = MPI_INFO_NULL;

  create(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(id(), mpi_comm, mpi_info);
#else
  ARCANE_UNUSED(pm);
  ARCANE_THROW(NotSupportedException,"HDF5 is not compiled with MPI support");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HProperty::
createDatasetTransfertCollectiveMPIIO()
{
#ifdef H5_HAVE_PARALLEL
  create(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(id(), H5FD_MPIO_COLLECTIVE);
#else
  ARCANE_THROW(NotSupportedException,"HDF5 is not compiled with MPI support");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardTypes::
StandardTypes()
{
  initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardTypes::
StandardTypes(bool do_init)
{
  if (do_init)
    initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardTypes::
initialize()
{
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_CHAR);
    m_char_id.setId(type_id);
    //H5Tset_precision(m_int_id,8*1);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_UCHAR);
    m_uchar_id.setId(type_id);
    //H5Tset_precision(m_int_id,8*1);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_SHORT);
    H5Tset_precision(type_id,8*sizeof(short));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_short_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_INT);
    H5Tset_precision(type_id,8*sizeof(int));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_int_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_LONG);
    H5Tset_precision(type_id,8*sizeof(long));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_long_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_USHORT);
    H5Tset_precision(type_id,8*sizeof(unsigned short));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_ushort_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_UINT);
    H5Tset_precision(type_id,8*sizeof(unsigned int));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_uint_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_ULONG);
    H5Tset_precision(type_id,8*sizeof(unsigned long));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_ulong_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_DOUBLE);
    H5Tset_precision(type_id,8*sizeof(double));
    H5Tset_order(type_id,H5T_ORDER_LE);
    m_real_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcreate(H5T_COMPOUND,sizeof(Real2POD));
    _H5Tinsert(type_id,"X",HOFFSET(Real2POD,x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"Y",HOFFSET(Real2POD,y),H5T_NATIVE_DOUBLE);
    m_real2_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcreate(H5T_COMPOUND,sizeof(Real3POD));
    _H5Tinsert(type_id,"X",HOFFSET(Real3POD,x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"Y",HOFFSET(Real3POD,y),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"Z",HOFFSET(Real3POD,z),H5T_NATIVE_DOUBLE);
    m_real3_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcreate(H5T_COMPOUND,sizeof(Real2x2POD));
    _H5Tinsert(type_id,"XX",HOFFSET(Real2x2POD,x.x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"XY",HOFFSET(Real2x2POD,x.y),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"YX",HOFFSET(Real2x2POD,y.x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"YY",HOFFSET(Real2x2POD,y.y),H5T_NATIVE_DOUBLE);
    m_real2x2_id.setId(type_id);
  }
  {
    hid_t type_id = H5Tcreate(H5T_COMPOUND,sizeof(Real3x3POD));
    _H5Tinsert(type_id,"XX",HOFFSET(Real3x3POD,x.x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"XY",HOFFSET(Real3x3POD,x.y),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"XZ",HOFFSET(Real3x3POD,x.z),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"YX",HOFFSET(Real3x3POD,y.x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"YY",HOFFSET(Real3x3POD,y.y),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"YZ",HOFFSET(Real3x3POD,y.z),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"ZX",HOFFSET(Real3x3POD,z.x),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"ZY",HOFFSET(Real3x3POD,z.y),H5T_NATIVE_DOUBLE);
    _H5Tinsert(type_id,"ZZ",HOFFSET(Real3x3POD,z.z),H5T_NATIVE_DOUBLE);
    m_real3x3_id.setId(type_id);
  }

  // HDF5 1.10 et 1.12 ne supportent pas encore les types 'BFloat16' et 'Float16'.
  // Lorsque ce sera le cas, on pourra utiliser le type fourni par HDF5.
  // (NOTE: HDF5 1.14.4 supporte Float16)

  // Ajoute type opaque pour BFloat16.
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_B16);
    m_bfloat16_id.setId(type_id);
  }
  // Ajoute type opaque pour Float16.
  {
    hid_t type_id = H5Tcopy(H5T_NATIVE_B16);
    m_float16_id.setId(type_id);
  }

 }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardTypes::
~StandardTypes()
{
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardTypes::
_H5Tinsert(hid_t type,const char* name,Integer offset,hid_t field_id)
{
  herr_t herr = H5Tinsert(type,name,offset,field_id);
  if (herr<0){
    ARCANE_FATAL("Can not insert type");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_REAL_NOT_BUILTIN
hid_t StandardTypes::
nativeType(Real) const
{
  ARCANE_FATAL("Real is a complex type");
}
#endif

hid_t StandardTypes::
saveType(eDataType sd) const
{
  switch(sd){
  case DT_Byte: return saveType(Byte());
  case DT_Real: return saveType(Real());
  case DT_Real2: return saveType(Real2());
  case DT_Real2x2: return saveType(Real2x2());
  case DT_Real3: return saveType(Real3());
  case DT_Real3x3: return saveType(Real3x3());
  case DT_Int8: return saveType(Int8());
  case DT_Int16: return saveType(Int16());
  case DT_Int32: return saveType(Int32());
  case DT_Int64: return saveType(Int64());
  case DT_Float32: return saveType(Float32());
  case DT_Float16: return saveType(Float16());
  case DT_BFloat16: return saveType(BFloat16());
  default:
    throw ArgumentException(String::format("Bad type '{0}'",sd));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

hid_t StandardTypes::
nativeType(eDataType sd) const
{
  switch(sd){
  case DT_Byte: return nativeType(Byte());
  case DT_Real: return nativeType(Real());
  case DT_Real2: return nativeType(Real2());
  case DT_Real2x2: return nativeType(Real2x2());
  case DT_Real3: return nativeType(Real3());
  case DT_Real3x3: return nativeType(Real3x3());
  case DT_Int8: return nativeType(Int8());
  case DT_Int16: return nativeType(Int16());
  case DT_Int32: return nativeType(Int32());
  case DT_Int64: return nativeType(Int64());
  case DT_Float32: return nativeType(Float32());
  case DT_Float16: return nativeType(Float16());
  case DT_BFloat16: return nativeType(BFloat16());
  default:
    throw ArgumentException(String::format("Bad type '{0}'",sd));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandardArray::
StandardArray(hid_t hfile,const String& hpath)
: m_hfile(hfile)
, m_hpath(hpath)
, m_ids_hpath(hpath + "_Ids")
, m_is_init(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardArray::
readDim()
{
  if (m_is_init)
    return;
  m_is_init = true;
  m_hdataset.open(m_hfile,m_hpath);
  HSpace hspace(m_hdataset.getSpace());
  {
    const int max_dim = 256; // Nombre maxi de dimensions des tableaux HDF
    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    int nb_dim = H5Sget_simple_extent_ndims(hspace.id());
    H5Sget_simple_extent_dims(hspace.id(),hdf_dims,max_dims);
    for( Integer i=0; i<nb_dim; ++i ){
      //cerr << "** DIM i=" << i << " hdim=" << hdf_dims[i]
      //   << " max=" << max_dims[i] << '\n';
      m_dimensions.add((Int64)hdf_dims[i]);
    }
  }
  // Vérifie s'il existe une variable suffixée '_Ids' contenant les numéros
  // uniques des entités
  m_ids_dataset.openIfExists(m_hfile,m_ids_hpath);
  //cout << "TRY OPEN ID DATASET path=" << m_ids_hpath << " r=" << m_ids_dataset.id()>0 << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardArray::
setIdsPath(const String& ids_path)
{
  m_ids_hpath = ids_path;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StandardArray::
_write(const void* buffer,Integer nb_element,hid_t save_type,hid_t native_type)
{
  if (!m_is_init){
    hsize_t dims[1];
    dims[0] = nb_element;
    
    HSpace hspace;
    hspace.createSimple(1,dims);
    if (hspace.isBad())
      ARCANE_THROW(IOException,"Can not create space");

    m_dimensions.clear();
    m_dimensions.add(nb_element);

    m_hdataset.recursiveCreate(m_hfile,m_hpath,save_type,hspace,H5P_DEFAULT);
    if (m_hdataset.isBad())
      ARCANE_THROW(IOException,"Can not create dataset");

    m_is_init = true;
  }

  m_hdataset.write(native_type,buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StandardArray::
exists() const
{
  HDataset dataset;
  dataset.openIfExists(m_hfile,m_hpath);
  return dataset.id()>0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> StandardArrayT<DataType>::
StandardArrayT(hid_t hfile,const String& hpath)
: StandardArray(hfile,hpath)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
read(StandardTypes& st,ArrayView<DataType> buffer)
{
  m_hdataset.readWithException(st.nativeType(DataType()),buffer.data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
directRead(StandardTypes& st,Array<DataType>& buffer)
{
  readDim();
  buffer.resize(m_dimensions[0]);
  read(st,buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
parallelRead(IParallelMng* pm,StandardTypes& st,Array<DataType>& buffer,Int64Array& unique_ids)
{
  bool is_master = pm->isMasterIO();
  Integer master_rank = pm->masterIORank();
  bool has_ids = false;
  if (is_master){
    read(st,buffer);
    Integer buf_size = buffer.size();
    if (m_ids_dataset.id()>0){
      has_ids = true;
      m_ids_dataset.read(st.nativeType(Int64()),unique_ids.data());
    }
    Integer infos[2];
    infos[0] = buf_size;
    infos[1] = has_ids ? 1 : 0;
    IntegerArrayView iav(2,infos);
    pm->broadcast(iav,master_rank);
    pm->broadcast(buffer,master_rank);
    pm->broadcast(unique_ids,master_rank);
  }
  else{
    Integer infos[2];
    IntegerArrayView iav(2,infos);
    pm->broadcast(iav,master_rank);
    Integer buf_size = infos[0];
    has_ids = infos[1]!=0;
    buffer.resize(buf_size);
    unique_ids.resize(buf_size);
    pm->broadcast(buffer,master_rank);
    pm->broadcast(unique_ids,master_rank);
  }
  if (!has_ids){
    for( Integer i=0, is=unique_ids.size(); i<is; ++i )
      unique_ids[i] = i;      
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
write(StandardTypes& st,ConstArrayView<DataType> buffer)
{
  Integer nb_element = buffer.size();
  _write(buffer.data(),nb_element,st.saveType(DataType()),st.nativeType(DataType()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
_writeSortedValues(ITraceMng* tm,StandardTypes& st,
                   ConstArrayView<DataType> buffer,
                   Int64ConstArrayView unique_ids)
{
  ARCANE_UNUSED(tm);

  Integer total_size = buffer.size();
  Integer nb_element = unique_ids.size();
  Integer dim2_size = 1;
  if (nb_element != total_size){
    if (nb_element == 0)
      ARCANE_THROW(ArgumentException,"unique_ids size is zero but not buffer size ({0})",
                   total_size);
    dim2_size = total_size / nb_element;
    if (dim2_size*nb_element != total_size)
      ARCANE_THROW(ArgumentException,"buffer size ({0}) is not a multiple of unique_ids size ({1})",
                   total_size,nb_element);
  }

  UniqueArray<ValueWithUid> values_to_sort(nb_element);
  UniqueArray<DataType> out_buffer(total_size);
  //tm->info() << " WRITE total_size=" << total_size
  //<< " uid_size=" << unique_ids.size();
  for( Integer i=0; i<nb_element; ++i ){
    values_to_sort[i].m_uid = unique_ids[i];
    values_to_sort[i].m_index = i;
    //values_to_sort[i].m_value = buffer[i];
    //tm->info() << "BEFORE SORT i=" << i << " uid=" << unique_ids[i];
  }
  std::sort(std::begin(values_to_sort),std::end(values_to_sort));
  for( Integer i=0; i<nb_element; ++i ){
    Integer old_index = values_to_sort[i].m_index;
    for( Integer j=0; j<dim2_size; ++j ){
      Integer pos = (i*dim2_size)+j;
      out_buffer[pos] = buffer[(old_index*dim2_size)+j];
      //tm->info() << "AFTER SORT i=" << i << " uid=" << values_to_sort[i].m_uid
      //           << " j=" << j << " pos=" << pos
      //           << " value=" << out_buffer[pos];
    }
  }
  write(st,out_buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void StandardArrayT<DataType>::
parallelWrite(IParallelMng* pm,StandardTypes& st,
              ConstArrayView<DataType> buffer,Int64ConstArrayView unique_ids)
{
  //TODO:
  // Pour l'instant, seul le proc maitre ecrit.
  // Il recupère toutes les valeurs, les trie par uniqueId croissant
  // et les écrit.
  // Il est important que tout soit trié par ordre croissant car
  // cela permet de garder le même ordre d'écriture même en présence
  // de repartitionnement du maillage. La relecture considère
  // que cette contrainte est respectée et ne relit les informations
  // des uniqueId qu'au démarrage du cas.
  bool is_parallel = pm->isParallel();
  ITraceMng* tm = pm->traceMng();

  if (!is_parallel){
    _writeSortedValues(tm,st,buffer,unique_ids);
    return;
  }

  bool is_master = pm->isMasterIO();
  Integer master_rank = pm->masterIORank();
  Integer nb_rank = pm->commSize();
  Integer buf_size = buffer.size();
  Integer unique_id_size = unique_ids.size();
  IntegerUniqueArray rank_sizes(2*nb_rank);
  // Le sous-domaine maitre récupère les infos de tous les autres.
  // Si un sous-domaine n'a pas d'éléments à envoyer, il ne fait rien
  // (on n'envoie pas de buffers vides)
  if (is_master){
    Integer buf[2];
    buf[0] = buf_size;
    buf[1] = unique_id_size;
    IntegerArrayView iav(2,buf);
    pm->allGather(iav,rank_sizes);

    Integer buffer_total_size = 0;
    Integer unique_id_total_size = 0;
    IntegerUniqueArray buffer_rank_index(nb_rank);
    IntegerUniqueArray unique_id_rank_index(nb_rank);
    
    for( Integer i=0; i<nb_rank; ++i ){
      buffer_rank_index[i] = buffer_total_size;
      buffer_total_size += rank_sizes[(i*2)];
      unique_id_rank_index[i] = unique_id_total_size;
      unique_id_total_size += rank_sizes[(i*2)+1];
    }
      
    UniqueArray<DataType> full_buffer(buffer_total_size);
    Int64UniqueArray full_unique_ids(unique_id_total_size);

    for( Integer i=0; i<nb_rank; ++i ){
      // Ne recoit pas de valeurs des processus n'ayant pas de valeurs
      if (rank_sizes[(i*2)]==0)
        continue;
      ArrayView<DataType> local_buf(rank_sizes[(i*2)],&full_buffer[ buffer_rank_index[i] ]);
      Int64ArrayView local_unique_ids(rank_sizes[(i*2)+1],&full_unique_ids[ unique_id_rank_index[i] ]);
      if (i==master_rank){
        local_buf.copy(buffer);
        local_unique_ids.copy(unique_ids);
      }
      else{
        pm->recv(local_buf,i);
        pm->recv(local_unique_ids,i);
      }
    }
    tm->info(5) << "PARALLEL WRITE path=" << m_hpath << " total_size=" << full_buffer.size();
    _writeSortedValues(tm,st,full_buffer,full_unique_ids);
  }
  else{
    Integer buf[2];
    buf[0] = buf_size;
    buf[1] = unique_id_size;
    IntegerArrayView iav(2,buf);
    pm->allGather(iav,rank_sizes);
    // Pas la peine d'envoyer des buffers vides
    if (buffer.size()>0){
      pm->send(buffer,master_rank);
      pm->send(unique_ids,master_rank);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class StandardArrayT<Real>;
template class StandardArrayT<Real3>;
template class StandardArrayT<Real3x3>;
template class StandardArrayT<Real2>;
template class StandardArrayT<Real2x2>;
template class StandardArrayT<Int16>;
template class StandardArrayT<Int32>;
template class StandardArrayT<Int64>;
template class StandardArrayT<Byte>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataType
StandardScalarT<DataType>::
read(Hdf5Utils::StandardTypes & st)
{
  Hdf5Utils::HDataset m_hdataset;
  m_hdataset.open(m_hfile,m_hpath);
  Hdf5Utils::HSpace hspace(m_hdataset.getSpace());
  {
    const int max_dim = 256; // Nombre maxi de dimensions des tableaux HDF
    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    int nb_dim = H5Sget_simple_extent_ndims(hspace.id());
    H5Sget_simple_extent_dims(hspace.id(),hdf_dims,max_dims);
    
    if (nb_dim!=1 || hdf_dims[0]!=1)
      ARCANE_THROW(IOException,"Cannot read non scalar");
  }
  
  DataType dummy;
  m_hdataset.read(st.nativeType(DataType()),&dummy);
  return dummy;
}

/*---------------------------------------------------------------------------*/

template<> String
StandardScalarT<String>::
read(Hdf5Utils::StandardTypes & st)
{
  ByteUniqueArray utf8_bytes;
  

  Hdf5Utils::HDataset m_hdataset;
  m_hdataset.open(m_hfile,m_hpath);
  Hdf5Utils::HSpace hspace(m_hdataset.getSpace());
  {
    const int max_dim = 256; // Nombre maxi de dimensions des tableaux HDF
    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    int nb_dim = H5Sget_simple_extent_ndims(hspace.id());
    H5Sget_simple_extent_dims(hspace.id(),hdf_dims,max_dims);
    
    if (nb_dim != 1)
      ARCANE_THROW(IOException,"Cannot read multidim string");
    utf8_bytes.resize(hdf_dims[0]);
  }
  
  m_hdataset.read(st.nativeType(Byte()),utf8_bytes.data());
  return String(utf8_bytes);
}

/*---------------------------------------------------------------------------*/

template<typename DataType> void
StandardScalarT<DataType>::
write(Hdf5Utils::StandardTypes & st, const DataType & t)
{
  hsize_t dims[1] = { 1 };
  Hdf5Utils::HSpace hspace;
  hspace.createSimple(1,dims);
  if (hspace.isBad())
    ARCANE_THROW(IOException,"Can not create space");

  Hdf5Utils::HDataset m_hdataset;
  m_hdataset.recursiveCreate(m_hfile,m_hpath,st.saveType(DataType()),hspace,H5P_DEFAULT);
  if (m_hdataset.isBad())
    ARCANE_THROW(IOException,"Can not create dataset");

  herr_t herr = m_hdataset.write(st.nativeType(DataType()),&t);
  if (herr<0)
    ARCANE_THROW(IOException,"Cannot write data");
}

/*---------------------------------------------------------------------------*/

template<> void
StandardScalarT<String>::
write(Hdf5Utils::StandardTypes & st, const String & s)
{
  ByteConstArrayView utf8_bytes = s.utf8();
  
  hsize_t dims[1];
  dims[0] = utf8_bytes.size() + 1;
  
  Hdf5Utils::HSpace hspace;
  hspace.createSimple(1,dims);
  if (hspace.isBad())
    ARCANE_THROW(IOException,"Can not create space");

  Hdf5Utils::HDataset m_hdataset;
  m_hdataset.recursiveCreate(m_hfile,m_hpath,st.saveType(Byte()),hspace,H5P_DEFAULT);
  if (m_hdataset.isBad())
    ARCANE_THROW(IOException,"Can not create dataset");

  herr_t herr = m_hdataset.write(st.nativeType(Byte()),utf8_bytes.data());
  if (herr<0)
    ARCANE_THROW(IOException,"Cannot write data");
}

/*---------------------------------------------------------------------------*/

template class StandardScalarT<Real>;
template class StandardScalarT<Real3>;
template class StandardScalarT<Real3x3>;
template class StandardScalarT<Real2>;
template class StandardScalarT<Real2x2>;
template class StandardScalarT<Int16>;
template class StandardScalarT<Int32>;
template class StandardScalarT<Int64>;
template class StandardScalarT<Byte>;
template class StandardScalarT<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Hdf5Utils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
