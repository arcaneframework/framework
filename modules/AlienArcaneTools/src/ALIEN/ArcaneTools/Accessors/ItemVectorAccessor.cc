#include "ALIEN/ArcaneTools/Accessors/ItemVectorAccessor.h"

/*---------------------------------------------------------------------------*/

#define ALIEN_ACCESSOR_ITEMVECTORACCESSOR_CC

// Si on est en debug ou qu'on ne souhaite pas l'inlining, VectorAccessorT est inclus ici
#ifdef ALIEN_INCLUDE_TEMPLATE_IN_CC
#include "ALIEN/ArcaneTools/Accessors/ItemVectorAccessorT.h"
#endif /* SHOULD_BE_INCLUDE_IN_CC */

#undef ALIEN_ACCESSOR_ITEMVECTORACCESSOR_CC

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  template class ALIEN_ARCANETOOLS_EXPORT ItemVectorAccessorT<double>;

#ifdef WIN32
  template class ALIEN_ARCANETOOLS_EXPORT ItemVectorAccessorT<double>::VectorElement;
  template void ALIEN_ARCANETOOLS_EXPORT
  ItemVectorAccessorT<double>::VectorElement::operator=(const double&);
#endif

  /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
