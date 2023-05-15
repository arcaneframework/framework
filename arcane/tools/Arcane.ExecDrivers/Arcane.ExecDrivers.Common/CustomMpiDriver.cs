using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using Arcane.ExecDrivers.Common;

namespace Arcane.ExecDrivers.Common
{
  // Informations de configuration.
  // Les champs de cette classe sont initialisés via JSON
  class Config
  {
    public string MpiDistrib { get; set; }
    public string MpiRoot { get; set; }
    public string MpiPrefix { get; set; }
  }

  class CustomMpiDriver : ICustomExecDriver
  {
    bool m_is_init;
    Config m_config;
    Assembly m_json_assembly;
    string m_config_path;

    public string Name { get { return "CustomMpiDriver"; } }

    void _Init()
    {
      if (m_is_init)
        return;
      // Avec '.Net Core 3.0', il faut charger explicitement la DLL
      // de 'NewtonSoft.Json' (car cette classe est chargée dynamiquement)
      {
        Assembly a = Assembly.GetExecutingAssembly();
        string assembly_path = Path.GetDirectoryName(a.Location);
        m_config_path = Path.Combine(assembly_path, "ArcaneCea.config");
        Console.WriteLine("LOCATION_PATH={0}", m_config_path);
        string json_assembly_path = Path.Combine(assembly_path,"Newtonsoft.Json.dll");
        Console.WriteLine("Trying loading newtonsoft.json={0}", json_assembly_path);
        m_json_assembly = Utils.LoadAssembly(json_assembly_path);
        Console.WriteLine("Newtonsoft? {0}",m_json_assembly!=null);
      }

      string host_name = Environment.MachineName;
      _ReadConfig();
      Console.WriteLine("TRY CUSTOM MPI_LAUNCHER machine={0} mpi_root={1}", host_name, m_config.MpiRoot);
      m_is_init = true;
    }

    void _ReadConfig()
    {
      // Lit le fichier de configuration 'ArcaneCea.config' au format JSON
      // qui doit se trouver dans le même répertoire que cette assembly.

      Newtonsoft.Json.JsonSerializer ser = new Newtonsoft.Json.JsonSerializer();
      using (StreamReader sr = new StreamReader(m_config_path)) {
        Newtonsoft.Json.JsonTextReader r = new Newtonsoft.Json.JsonTextReader(sr);
        m_config = (Config)ser.Deserialize(r, typeof(Config));
      }
    }

    public bool HandleMpiLauncher(Arcane.ExecDrivers.Common.ExecDriverProperties p, string mpi_launcher)
    {
      _Init();
      Console.WriteLine("TRY CUSTOM MPI_LAUNCHER name={0}", mpi_launcher);
      Console.WriteLine("OS_VERSION={0}", Environment.OSVersion.VersionString);
      if (mpi_launcher.EndsWith("ccc_mprun", StringComparison.Ordinal))
        return _HandleCCCMprun(p);
      if (mpi_launcher.EndsWith("mpiexec", StringComparison.Ordinal))
        return _HandleMpiExec(p);
      return false;
    }

    bool _HandleMpiExec(ExecDriverProperties p)
    {
      Console.WriteLine("Handle Mpiexec");
      return false;
    }

    // Gère le lancement via 'ccc_mprun'
    bool _HandleCCCMprun(ExecDriverProperties p)
    {
      Console.WriteLine("Handling 'ccc_mprun'");

      // Si on est dans un job slurm, on suppose qu'on est sur un noeud de l'allocation
      // (ce n'est pas forcément le cas. Pour être sur, il faudrait regarder si le noeud
      // sur lequel on tourne est dans la liste des noeuds de la variable d'environnement
      // SLURM_JOB_NODELIST.
      string job_id = Utils.GetEnvironmentVariable("SLURM_JOB_ID");
      bool is_already_allocated = !string.IsNullOrEmpty(job_id);
      Console.WriteLine("is_already_allocated={0}", is_already_allocated);

      if (is_already_allocated) {
        // Avec ccc_mprun, il faut ajouter au srun l'option '--exclusive' pour ne pas
        // avoir de problèmes de cohérence sur le nombre de processus à utiliser.
        p.MpiLauncherArgs.Add("-E");
        p.MpiLauncherArgs.Add("--exclusive");
      }

      if (p.UseTotalview) {
        p.MpiLauncherArgs.Add("-d tv");
        p.UseTotalview = false;
      }

      int nb_thread = p.NbTaskPerProcess * p.NbSharedMemorySubDomain;
      Console.WriteLine("NB_THREAD={0} ({1},{2})", nb_thread, p.NbSharedMemorySubDomain, p.NbTaskPerProcess);

      p.MpiLauncherArgs.Add("-n");
      p.MpiLauncherArgs.Add(p.NbProc.ToString());

      if (!is_already_allocated) {
        p.MpiLauncherArgs.AddRange(new string[] { "-x", "-v", });
      }
      // Alloue les coeurs suivant le nombre de threads demandes
      // et specifie un nombre de noeuds pour etre sur que les threads seront bien repartis
      if (nb_thread > 1) {
        int nb_thread_power_2 = 1;
        while (nb_thread_power_2 < nb_thread)
          nb_thread_power_2 *= 2;
        p.MpiLauncherArgs.Add("-c" + nb_thread_power_2);
        //Utils.SetEnvironmentVariable("ARCANE_BIND_THREADS", "1");
        //Utils.SetEnvironmentVariable("ARCANE_SPINLOCK_BARRIER", "1");
      }

      return true;
    }
  }
}
