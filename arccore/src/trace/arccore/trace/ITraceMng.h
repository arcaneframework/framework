// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITraceMng.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des traces.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_ITRACEMNG_H
#define ARCCORE_TRACE_ITRACEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/trace/TraceMessage.h"
#include "arccore/base/RefDeclarations.h"

#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments de ITraceMessageListener::visitMessage().
 *
 * \a buffer() contient la chaîne de caractère à afficher.
 * \a buffer() se termine toujours par un zéro terminal.
 *
 * Une instance de cette classe est un objet temporaire qui ne doit
 * pas être conservé au delà de l'appel ITraceMessageListener::visitMessage().
 */
class ARCCORE_TRACE_EXPORT TraceMessageListenerArgs
{
 public:
  TraceMessageListenerArgs(const TraceMessage* msg,ConstArrayView<char> buf)
  : m_message(msg), m_buffer(buf){}
 public:
  //! Infos sur le message de trace
  const TraceMessage* message() const
  {
    return m_message;
  }

  //! Chaîne de caractères du message.
  ConstArrayView<char> buffer() const
  {
    return m_buffer;
  }
 private:
  const TraceMessage* m_message;
  ConstArrayView<char> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un visiteur pour les messages de trace.
 */
class ARCCORE_TRACE_EXPORT ITraceMessageListener
{
 public:
  virtual ~ITraceMessageListener() {}
  /*!
   * \brief Réception du message \a msg contenant la châine \a str.
   *
   * Si le retour est \a true, le message n'est pas utilisé par le ITraceMng.
   *
   * L'instance a le droit d'appeler ITraceMng::writeDirect() pour écrire
   * directement des messages pendant l'appel à cette méthode.
   *
   * \warning Attention, cette fonction doit être thread-safe car elle peut être
   * appelée simultanément par plusieurs threads.
   */
  virtual bool visitMessage(const TraceMessageListenerArgs& args) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Flux pour une trace.
 *
 * Cette instance utilise un compteur de référence et peut être manipulée
 * via une instance de ReferenceCounter.
 */
class ARCCORE_TRACE_EXPORT ITraceStream
{
 public:
  typedef ReferenceCounterTag ReferenceCounterTagType;
 public:
  virtual ~ITraceStream() = default;
 public:
  //! Ajoute une référence.
  virtual void addReference() =0;
  //! Supprime une référence.
  virtual void removeReference() =0;
  //! Flux standard associé. Peut retourner nul.
  virtual std::ostream* stream() =0;
 public:
  static ITraceStream* createFileStream(const String& filename);
  static ITraceStream* createStream(std::ostream* stream,bool need_destroy);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de traces.
 *
 Une instance de cette classe gère les flots de traces.
 Pour envoyer un message, il suffit d'appeler la méthode corrrespondante
 (info() pour un message d'information, error() pour une erreur, ...)
 pour récupérer un flot et d'utiliser l'opérateur << sur ce flot pour
 transmettre un message.
 
 Par exemple:
 \code
 ITraceMng* tr = ...;
 tr->info() << "Ceci est une information.";
 int proc_id = 0;
 tr->error() << "Erreur sur le processeur " << proc_id;
 \endcode

 Le message est envoyé lors de la destruction du flot. Dans les
 exemples précédents, les flots sont temporairement créés (par la méthode info())
 et détruits dès que l'opérateur << a été appliqué dessus.

 \warning Il faut absolument appeler la méthode finishInitialize() avant
 d'utiliser les appels aux méthodes pushTraceClass() et popTraceClass().
 
 Si on souhaite envoyer un message en plusieurs fois, il faut stocker
 le flot retourné:

 \code
 TraceMessage info = m_trace_mng->info();
 info() << "Début de l'information.\n"
 info() << "Fin de l'information.";
 \endcode

 Il est possible d'utiliser des formatteurs simples sur les messages
 (via la classe #TraceMessage)
 ou des formatteurs standards des iostream en appliquant l'opérateur operator()
 de TraceMessage.

 Les instances de cette classe sont gérées par compteur de référence. Il
 est préférable de conserver les instances dans un ReferenceCounter.
 */
class ARCCORE_TRACE_EXPORT ITraceMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
 public:
  virtual ~ITraceMng() = default;
 public:
  //! Flot pour un message d'erreur
  virtual TraceMessage error() =0;
  //! Flot pour un message d'erreur parallèle
  virtual TraceMessage perror() =0;
  //! Flot pour un message d'erreur fatale
  virtual TraceMessage fatal() =0;
  //! Flot pour un message d'erreur fatale parallèle
  virtual TraceMessage pfatal() =0;
  //! Flot pour un message d'avertissement
  virtual TraceMessage warning() =0;
  //! Flot pour un message d'avertissement parallèle
  virtual TraceMessage pwarning() =0;
  //! Flot pour un message d'information
  virtual TraceMessage info() =0;
  //! Flot pour un message d'information parallèle
  virtual TraceMessage pinfo() =0;
  //! Flot pour un message d'information d'une catégorie donnée
  virtual TraceMessage info(char category) =0;
  //! Flot pour un message d'information d'un niveau donné
  virtual TraceMessage info(Int32 level) =0;
  //! Flot pour un message d'information parallèle d'une catégorie donnée
  virtual TraceMessage pinfo(char category) =0;
  //! Flot pour un message d'information conditionnel.
  virtual TraceMessage info(bool) =0;
  //! Flot pour un message de log.
  virtual TraceMessage log() =0;
  //! Flot pour un message de log parallèle.
  virtual TraceMessage plog() =0;
  //! Flot pour un message de log précédé de l'heure.
  virtual TraceMessage logdate() =0;
  //! Flot pour un message de debug.
  virtual TraceMessageDbg debug(Trace::eDebugLevel =Trace::Medium) =0;
  //! Flot pour un message non utilisé
  virtual TraceMessage devNull() =0;

  //! \deprecated Utiliser setInfoActivated() à la place
  ARCCORE_DEPRECATED_2018 virtual bool setActivated(bool v) { return setInfoActivated(v); }
  /*!
   * \brief Modifie l'état d'activation des messages d'info.
   *
   * \return l'ancien état d'activation.
   */
  virtual bool setInfoActivated(bool v) =0;
  //! Indique si les sorties des messages d'informations sont activées.
  virtual bool isInfoActivated() const =0;

  //! Termine l'initialisation
  virtual void finishInitialize() =0;

  /*!
   * \brief Ajoute la classe \a s à la pile des classes de messages actifs.
   * \threadsafe
   */
  virtual void pushTraceClass(const String& name) =0;
	
  /*!
   * \brief Supprime la dernière classe de message de la pile.
   * \threadsafe
   */
  virtual void popTraceClass() =0;
	
  //! Flush tous les flots.
  virtual void flush() =0;

  /*!
   * \brief Redirige tous les messages sur le flot \a o.
   * \deprecated Utiliser la surcharge setRedirectStream(ITraceStream*).
   */
  ARCCORE_DEPRECATED_2018 virtual void setRedirectStream(std::ostream* o) =0;

  //! Redirige tous les messages sur le flot \a o
  virtual void setRedirectStream(ITraceStream* o) =0;

  //! Retourne le niveau dbg du fichier de configuration
  virtual Trace::eDebugLevel configDbgLevel() const =0;

 public:

  /*!
   * \brief Ajoute l'observateur \a v à ce gestionnaire de message.
   *
   * L'appelant reste propriétaire de \a v et doit l'enlever
   * via removeListener() avant de le détruire.
   */
  virtual void addListener(ITraceMessageListener* v) =0;

  //! Supprime l'observateur \a v de ce gestionnaire de message.
  virtual void removeListener(ITraceMessageListener* v) =0;

  /*!
   * \brief Positionne l'identifiant du gestionnnaire.
   *
   * Si non nul, l'identifiant est affiché en cas d'erreur pour
   * identifier l'instance qui affiche le message. L'identifiant
   * peut être quelconque. Par défaut, il s'agit du rang du processus et
   * du nom de la machine.
   */
  virtual void setTraceId(const String& id) =0;

  //! Identifiant du gestionnnaire.
  virtual const String& traceId() const =0;

  /*!
   * \brief Positionne le nom du fichier d'erreur à \a file_name.
   *
   * Si un fichier d'erreur est déjà ouvert, il est refermé et un nouveau
   * avec ce nouveau nom de fichier sera créé lors de la prochaine erreur.
   *
   * Si \a file_name est la chaîne nulle, aucun fichier d'erreur n'est utilisé.
   */
  virtual void setErrorFileName(const String& file_name) =0;

  /*!
   * \brief Positionne le nom du fichier de log à \a file_name.
   *
   * Si un fichier de log est déjà ouvert, il est refermé et un nouveau
   * avec ce nouveau nom de fichier sera créé lors du prochaine log.
   *
   * Si \a file_name est la chaîne nulle, aucun fichier de log n'est utilisé.
   */
  virtual void setLogFileName(const String& file_name) =0;

 public:

  //! Signale un début d'écriture du message \a message
  virtual void beginTrace(const TraceMessage* message) =0;

  //! Signale une fin d'écriture du message \a message
  virtual void endTrace(const TraceMessage* message) =0;

  /*!
   * \brief Envoie directement un message de type \a type.
   *
   * \a type doit correspondre à Trace::eMessageType.
   * Cette méthode ne doit être utilisée que par le wrapping .NET.
   */
  virtual void putTrace(const String& message,int type) =0;

 public:
  
  //! Positionne la configuration pour la classe de message \a name
  virtual void setClassConfig(const String& name,const TraceClassConfig& config) =0;

  //! Configuration associées à la classe de message \a name
  virtual TraceClassConfig classConfig(const String& name) const =0;

  /*!
   * \brief Positionne l'état 'maitre' de l'instance.
   *
   * Les instances qui ont cet attribut à \a true affichent
   * les messages sur les std::cout ainsi que les messages
   * perror() et pwarning(). Il est donc préférable qu'il n'y ait
   * qu'une seule instance de ITraceMng maître.
   */
  virtual void setMaster(bool is_master) =0;

  // Indique si l'instance est maître.
  virtual bool isMaster() const =0;

  /*!
   * \brief Positionne le niveau de verbosité des sorties.
   *
   * Les messages de niveau supérieur à ce niveau ne sont pas sortis.
   * Le niveau utilisé est celui donné en argument de info(Int32).
   * Le niveau par défaut est celui donné par TraceMessage::DEFAULT_LEVEL.
   */
  virtual void setVerbosityLevel(Int32 level) =0;

  //! Niveau de verbosité des messages
  virtual Int32 verbosityLevel() const =0;

  /*!
   * \brief Positionne le niveau de verbosité des sorties sur std::cout
   *
   * Cette propriété n'est utilisée que si isMaster() est vrai et
   * qu'on a redirigé les sorties listings. Sinon, c'est la propriété
   * verbosityLevel() qui est utilisée.
   */
  virtual void setStandardOutputVerbosityLevel(Int32 level) =0;

  //! Niveau de verbosité des messages sur std::cout
  virtual Int32 standardOutputVerbosityLevel() const =0;

  /*!
   * \internal
   * Indique que le gestionnaire de thread à changé et qu'il faut
   * redéclarer les structures gérant le multi-threading.
   * Interne à Arccore, ne pas utiliser.
   */
  virtual void resetThreadStatus() =0;

  /*!
   * \brief Écrit directement un message.
   *
   * Ecrit directement le message \a msg contenant la chaîne \a buf_array.
   * Le message n'est pas analysé par l'instance et est toujours écrit
   * sans aucun formattage spécifique. Cette opération ne doit en principe
   * être utilisée que par un ITraceMessageListener. Pour les autres cas,
   * il faut utiliser les traces standards.
   */
  virtual void writeDirect(const TraceMessage* msg,const String& str) =0;

  //! Supprime toutes les classes de configuration positionnées via setClassConfig().
  virtual void removeAllClassConfig() =0;

  /*!
   * \biref Applique le fonctor \a functor sur l'ensemble des TraceClassConfig enregistrés.
   *
   * Le premier argument de la paire est le nom de classe de configuration et
   * le deuxième sa valeur telle que retournée par classConfig().
   *
   * Il est permis de modifier le TraceClassConfig en cours de visite via
   * un appel à setClassConfig().
   */
  virtual void visitClassConfigs(IFunctorWithArgumentT<std::pair<String,TraceClassConfig>>* functor) =0;


 public:

  /*!
   * \brief Effectue un fatal() sur un message déjà fabriqué.
   *
   * Cette méthode permet d'écrire un code équivalent à:
   *
   * \code
   * fatal() << "MyMessage";
   * \endcode
   *
   * comme ceci:
   *
   * \code
   * fatalMessage(StandaloneTraceMessage{} << "MyMessage");
   * \endcode
   *
   * Cette deuxième solution permet de signaler au compilateur que
   * la méthode ne retournera pas et ainsi d'éviter certains avertissements
   * de compilation.
   */
  void fatalMessage [[noreturn]] (const StandaloneTraceMessage& o);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_TRACE_EXPORT
ITraceMng* arccoreCreateDefaultTraceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::arccoreCreateDefaultTraceMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
