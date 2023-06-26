#include <algorithm>
#include <arcane/utils/FatalErrorException.h>
#include <UserObjects/IndexManager/ArcGeoSim/ContactAbstractFamily.h>

#include "ArcGeoSim/Utils/ArrayUtils.h"
#include "ArcGeoSim/Utils/VMap.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BEGIN_NAMESPACE(ArcaneTools)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ContactAbstractFamily::ContactAbstractFamily(
    const ContactFamily& family, IIndexManager* manager)
: m_family(family)
, m_manager(manager)
{
  ;
}

/*---------------------------------------------------------------------------*/

ContactAbstractFamily::~ContactAbstractFamily()
{
  m_manager->keepAlive(this);
}

/*---------------------------------------------------------------------------*/

void
ContactAbstractFamily::uniqueIdToLocalId(
    Int32ArrayView localIds, Int64ConstArrayView uniqueIds) const
{
  ARCANE_ASSERT(
      (localIds.size() == uniqueIds.size()), ("Incompatible array argument sizes"));

  const Integer family_size = m_family.maxLocalId();
  VMap<Int64, Int32> uid_to_lid_map(family_size);

  const ContactInternal* internals = m_family.contactsInternal();
  for (Integer i = 0; i < family_size; ++i) {
    const Int64 uid = _form_unique_id(internals[i]);
    const Int32 lid = i;
    uid_to_lid_map[uid] = lid;
  }

  const Integer sample_size = uniqueIds.size();
  for (Integer i = 0; i < sample_size; ++i) {
    VMap<Int64, Int32>::const_iterator finder = uid_to_lid_map.find(uniqueIds[i]);
    if (finder == uid_to_lid_map.end())
      throw Arcane::FatalErrorException(A_FUNCINFO, "Cannot find contact uid");
    localIds[i] = finder.value();
  }
}

/*---------------------------------------------------------------------------*/

IIndexManager::IAbstractFamily::Item
ContactAbstractFamily::item(Int32 localId) const
{
  ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
  const ContactInternal& internal = m_family.contactsInternal()[localId];
  return IAbstractFamily::Item(_form_unique_id(internal), internal.owner());
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Integer>
ContactAbstractFamily::owners(Int32ConstArrayView localIds) const
{
  const Integer size = localIds.size();
  Array<Integer> result(size);
  const ContactInternal* internals = m_family.contactsInternal();
  for (Integer i = 0; i < size; ++i) {
    Int32 localId = localIds[i];
    ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
    result[i] = internals[localId].owner();
  }
  return result;
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Int64>
ContactAbstractFamily::uids(Int32ConstArrayView localIds) const
{
  const Integer size = localIds.size();
  Array<Int64> result(size);
  const ContactInternal* internals = m_family.contactsInternal();
  for (Integer i = 0; i < size; ++i) {
    Int32 localId = localIds[i];
    ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
    result[i] = _form_unique_id(internals[localId]);
  }
  return result;
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Int32>
ContactAbstractFamily::allLocalIds() const
{
  const Integer size = m_family.maxLocalId();
  Array<Int32> local_ids(size);
  for (Integer i = 0; i < size; ++i)
    local_ids[i] = i;
  return local_ids;
}

/*---------------------------------------------------------------------------*/

Int64
ContactAbstractFamily::_form_unique_id(const ContactInternal& contact)
{
  const Arcane::Item item_1(contact.item1Internal());
  const Arcane::Item item_2(contact.item2Internal());

  if (item_1.null()) {
    ARCANE_ASSERT((not item_2.null()), ("Inconsistent (null,null) contact"));
    const Int64 item_2_uid = item_2.uniqueId();
    ARCANE_ASSERT(((item_2_uid & ~m_integer_mask) == 0 and item_2_uid < m_integer_mask),
        ("Too large item2 unique id %d", item_2_uid));
    return item_2_uid << m_integer_size | m_integer_mask;
  } else {
    const Int64 item_1_uid = item_1.uniqueId();
    ARCANE_ASSERT(((item_1_uid & ~m_integer_mask) == 0 and item_1_uid < m_integer_mask),
        ("Too large item1 unique id %d", item_1_uid));

    if (item_2.null()) {
      return m_integer_mask << m_integer_size | item_1_uid;
    } else {
      const Int64 item_2_uid = item_2.uniqueId();
      ARCANE_ASSERT(((item_2_uid & ~m_integer_mask and item_2_uid < m_integer_mask) == 0),
          ("Too large item2 unique id %d", item_2_uid));
      return item_2_uid << m_integer_size | item_1_uid;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
