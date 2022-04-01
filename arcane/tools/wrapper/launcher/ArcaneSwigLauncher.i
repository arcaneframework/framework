%module(directors="1") ArcaneLauncherModule

%import core/ArcaneSwigCore.i

%{
#include "ArcaneSwigUtils.h"
#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/launcher/internal/DirectExecutionFunctor.h"
using namespace Arcane;
%}

#define ARCANE_LAUNCHER_EXPORT

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%ignore Arcane::ArcaneLauncher::run;
%ignore Arcane::ArcaneLauncher::runDirect;
%ignore Arcane::ArcaneLauncher::setCommandLineArguments;
%ignore Arcane::ArcaneLauncher::createStandaloneAcceleratorMng;
%rename Arcane::ArcaneLauncher ArcaneLauncher_INTERNAL;

%feature("director") Arcane::IDirectExecutionFunctor;
%feature("director") Arcane::IDirectSubDomainExecutionFunctor;

ARCANE_STD_EXHANDLER
%include arcane/launcher/DirectExecutionContext.h
%include arcane/launcher/DirectSubDomainExecutionContext.h
%include arcane/launcher/ArcaneLauncher.h
%include arcane/launcher/internal/DirectExecutionFunctor.h
%exception;
