// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "Mesh/Geometry/Impl/GeometryTemplatesT.h"

template<typename GeometryT>
void 
GeometryServiceBase::
updateGroup(ItemGroup group, GeometryT & geometry) 
{
  PropertyMap::iterator igroup = m_group_property_map.find(group.internal());
  if (igroup == m_group_property_map.end())
    throw FatalErrorException(A_FUNCINFO,"Undefined ItemGroup property");

  ItemGroupGeometryProperty & properties = igroup->second;
  if ((properties.defined & ~properties.computed & ~properties.delegated) == 0)
    {
      traceMng()->debug() << "Group " << group.name() << " properties already done";
      return;
    }

  for(typename ItemGroupGeometryProperty::StorageInfos::iterator i = properties.storages.begin(); i != properties.storages.end(); ++i) 
    {
      IGeometryProperty::eProperty property = i->first;
      ItemGroupGeometryProperty::StorageInfo & storage = i->second;

      if ((properties.computed & property) or (properties.delegated & property))
        {
          traceMng()->debug() << "Property " << IGeometryProperty::name(property) << " is delayed for group " << group.name();
          continue; // skip that property
        }
      else        
        {
          traceMng()->debug() << "Property " << IGeometryProperty::name(property) << " will be computed for group " << group.name();
        }

      if (IGeometryProperty::isScalar(property)) 
        {
          boost::shared_ptr<RealVariable> & ivar = storage.realVar;
          if (not ivar and (storage.storageType & IGeometryProperty::PVariable))
            {
              String varName = IGeometryProperty::name(property)+String("Of")+group.name()+m_suffix;
              traceMng()->debug() << "Building Variable " << varName;
              ivar.reset(new RealVariable(VariableBuildInfo(group.mesh(),
                                                            varName,
                                                            group.itemFamily()->name(),
                                                            IVariable::PPrivate),
                                          group.itemKind()));
            }            

          boost::shared_ptr<RealGroupMap> & imap = storage.realMap;
          if (not imap and (storage.storageType & IGeometryProperty::PItemGroupMap))
            {
              imap.reset(new RealGroupMap(group));
              traceMng()->debug() << "Building Map " << imap->name();
            }
        }
      else
        {
          ARCANE_ASSERT((IGeometryProperty::isVectorial(property)),("Vectorial property expected"));
          boost::shared_ptr<Real3Variable> & ivar = storage.real3Var;
          if (not ivar and (storage.storageType & IGeometryProperty::PVariable))
            {
              String varName = IGeometryProperty::name(property)+String("Of")+group.name()+m_suffix;
              traceMng()->debug() << "Building Variable " << varName;
              ivar.reset(new Real3Variable(VariableBuildInfo(group.mesh(),
                                                             varName,
                                                             group.itemFamily()->name(),
                                                             IVariable::PPrivate),
                                           group.itemKind()));
            }            

          boost::shared_ptr<Real3GroupMap> & imap = storage.real3Map;
          if (not imap and (storage.storageType & IGeometryProperty::PItemGroupMap))
            {
              imap.reset(new Real3GroupMap(group));
              traceMng()->debug() << "Building Map " << imap->name();
            }
        }
    }

  if (properties.defined & ~properties.computed & ~properties.delegated) {
    GenericGSInternalUpdater<GeometryT> updater(geometry, traceMng());
    updater.setGroupProperty(&properties);
    group.applyOperation(&updater);
  }

  /*** L'affectation générale est désormais faite dans GeometryTemplatesT ***/
  for(typename ItemGroupGeometryProperty::StorageInfos::iterator i = properties.storages.begin(); i != properties.storages.end(); ++i) 
    {
      IGeometryProperty::eProperty property = i->first;
      if ((properties.computed & property) or (properties.delegated & property))
        continue;
      properties.computed |= property;
    }
}
