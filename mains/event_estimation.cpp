#include <vector>
#include <iostream>
#include <nuSQuIDS/nuSQuIDS.h>
#include "nusquids_LV.h"
#include "get_eff_area.h"
#include <gsl/gsl_integration.h>

// chris weaver tools
#include <PhysTools/histogram.h>
#include "weighting.h"
#include "autodiff.h"
#include "likelihood.h"

using namespace nusquids;

///Simple fit result structure
struct fitResult{
  std::vector<double> params;
  double likelihood;
  unsigned int nEval, nGrad;
  bool succeeded;
};

///Simple event structure
class Event{
  public:
    // public members
    double energy_proxy;
    double energy_neutrino;
    double costh;
    double conv_kaon_event;
    double conv_pion_event;
    double prompt_event;
    double astro_event;
    unsigned int year;

    Event():
      energy_proxy(std::numeric_limits<double>::quiet_NaN()),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(std::numeric_limits<double>::quiet_NaN()),
      conv_pion_event(std::numeric_limits<double>::quiet_NaN()),
      conv_kaon_event(std::numeric_limits<double>::quiet_NaN()),
      prompt_event(std::numeric_limits<double>::quiet_NaN()),
      astro_event(std::numeric_limits<double>::quiet_NaN()),
      year(0)
    {}

    Event(double energy_proxy,double costh,unsigned int year):
      energy_proxy(energy_proxy),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(costh),
      conv_pion_event(std::numeric_limits<double>::quiet_NaN()),
      conv_kaon_event(std::numeric_limits<double>::quiet_NaN()),
      prompt_event(std::numeric_limits<double>::quiet_NaN()),
      astro_event(std::numeric_limits<double>::quiet_NaN()),
      year(year)
    {}

    Event(double energy_proxy,double costh,unsigned int year, double conv_kaon_event, double conv_pion_event):
      energy_proxy(energy_proxy),
      energy_neutrino(std::numeric_limits<double>::quiet_NaN()),
      costh(costh),
      conv_pion_event(conv_pion_event),
      conv_kaon_event(conv_kaon_event),
      prompt_event(0),
      astro_event(0),
      year(year)
    {}
};

///Maximize a likelihood using the LBFGSB minimization algorithm
///\param likelihood The likelihood to maximize
///\param seed The seed values for all likelihood parameters
///\param indicesToFix The indices of likelihood parameters to hold constant at their seed va>
template<typename LikelihoodType>
fitResult doFitLBFGSB(LikelihoodType& likelihood, const std::vector<double>& seed,
            std::vector<unsigned int> indicesToFix={}){
  using namespace likelihood;

  LBFGSB_Driver minimizer;
  //minimizer.addParameter(SEED, MAGIC_NUMBER, MINIMUM, MAXIMUM);
  minimizer.addParameter(seed[0],.01,0.0); // if the normalization is allow to be zero it all goes crazy.
  minimizer.addParameter(seed[1],.01,0.0);
  minimizer.addParameter(seed[2],.01,0.0);

  for(auto idx : indicesToFix)
    minimizer.fixParameter(idx);

  minimizer.setChangeTolerance(1e-5);
  minimizer.setHistorySize(20);

  fitResult result;
  result.succeeded=minimizer.minimize(BFGS_Function<LikelihoodType>(likelihood));
  result.likelihood=minimizer.minimumValue();
  result.params=minimizer.minimumPosition();
  result.nEval=minimizer.numberOfEvaluations();
  result.nGrad=minimizer.numberOfEvaluations();

  return(result);
}


template<typename ContainerType, typename HistType, typename BinnerType>
void bin(const ContainerType& data, HistType& hist, const BinnerType& binner){
  for(const Event& event : data)
    binner(hist,event);
}

template<typename FunctionType>
double integrate(FunctionType f, double a, double b){
    double (*wrapper)(double,void*)=[](double x, void* params){
        FunctionType& f=*static_cast<FunctionType*>(params);
        return(f(x));
    };

    gsl_integration_workspace* ws=gsl_integration_workspace_alloc(5000);
    double result, error;
    gsl_function F;
    F.function = wrapper;
    F.params = &f;

    gsl_integration_qags(&F, a, b, 0, 1e-7, 5000, ws, &result, &error);
    gsl_integration_workspace_free(ws);

    return(result);
}

double GetAveragedFlux(nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh, double enu_min, double enu_max) {
    if(enu_min > enu_max)
      throw std::runtime_error("Min energy in the bin larger than large energy in the bin.");
    if (enu_min >= 1.0e6)
      return 0.;
    if (enu_max <= 1.0e2)
      return 0.;
    double GeV = 1.0e9;
    if (flavor == NUMU){
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,0));},enu_min,enu_max)/(enu_max-enu_min);
    } else {
        return integrate([&](double enu){return(nus->EvalFlavor(1,costh,enu*GeV,1));},enu_min,enu_max)/(enu_max-enu_min);
    }
}

double GetAveragedFlux(nuSQUIDSAtm<nuSQUIDSLV> * nus,PTypes flavor, double costh_min, double costh_max, double enu_min, double enu_max) {
  return (GetAveragedFlux(nus,flavor,costh_max,enu_min,enu_max) + GetAveragedFlux(nus,flavor,costh_min,enu_min,enu_max))/std::abs(costh_max-costh_min);
}

//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//
//==================== NUISANCE PARAMETERS REWEIGHTERS =========================//


template<typename T>
struct powerlawWeighter : public GenericWeighter<powerlawWeighter<T>>{
private:
	T index;
public:
	using result_type=T;
	powerlawWeighter(T i):index(i){}
	
	template<typename Event>
	result_type operator()(const Event& e) const{
		return(pow((double)e.primaryEnergy,index));
	}
};

//Tilt a spectrum by an incremental powerlaw index about a precomputed median energy
template<typename Event, typename T>
struct powerlawTiltWeighter : public GenericWeighter<powerlawTiltWeighter<Event,T>>{
private:
	double medianEnergy;
	T deltaIndex;
	//typename Event::crTiltValues Event::* cachedData;
public:
	using result_type=T;
	powerlawTiltWeighter(double me, T dg/*, typename Event::crTiltValues Event::* c*/):
	medianEnergy(me),deltaIndex(dg)/*,cachedData(c)*/{}
	
	result_type operator()(const Event& e) const{
		//const typename Event::crTiltValues& cache=e.*cachedData;
		result_type weight=pow(e.energy_proxy/medianEnergy,-deltaIndex);
		return(weight);
	}
};

template<typename T, typename Event, typename U>
struct cachedValueWeighter : public GenericWeighter<cachedValueWeighter<T,Event,U>>{
private:
	U Event::* cachedPtr;
public:
	using result_type=T;
	cachedValueWeighter(U Event::* ptr):cachedPtr(ptr){}
	result_type operator()(const Event& e) const{
		return(result_type(e.*cachedPtr));
	}
};

struct DiffuseFitWeighterMaker{
private:
	static constexpr double medianConvEnergy=2020;
	//static constexpr double medianPromptEnergy=7887;
public:
	DiffuseFitWeighterMaker()
	{}

	template<typename DataType>
	std::function<DataType(const Event&)> operator()(const std::vector<DataType>& params) const{
    // check that we are getting the right number of nuisance parameters
		assert(params.size()==3);
		//unpack things so we have legible names
		DataType convNorm=params[0];
		DataType CRDeltaGamma=params[1];
		DataType piKRatio=params[2];

		using cachedWeighter=cachedValueWeighter<DataType,Event,double>;
		cachedWeighter convPionFlux(&Event::conv_pion_event); // we get the pion component
		cachedWeighter convKaonFlux(&Event::conv_kaon_event); // we get the kaon component

		auto conventionalComponent = convNorm*(convPionFlux + piKRatio*convKaonFlux)
		                             *powerlawTiltWeighter<Event,DataType>(medianConvEnergy, CRDeltaGamma); // we sum them upp according to some pi/k ratio.

    /*
     * We will deal with the prompt and astrophysical components later. CA
		auto promptComponent = promptNorm*promptFlux
		                       *powerlawTiltWeighter<Event,DataType>(medianPromptEnergy, CRDeltaGamma);

    */

		return (conventionalComponent);
	}
};

//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//
//===================MAIN======================================================//


int main(int argc, char** argv)
{
    if(argc < 6){
        std::cout << "Invalid number of arguments. The arguments should be given as follows: \n"
                     "1) Path to the effective area hdf5.\n"
                     "2) Path to the observed events file.\n"
                     "3) Path to the kaon component nusquids calculated flux.\n"
                     "4) Path to the pion component nusquids calculated flux.\n"
                     "5) [optional] Path to output the event expectations.\n"
                     << std::endl;
        exit(1);
    }

    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//
    bool quiet = false;
    size_t evalThreads=1;
    std::vector<double> fitSeed {1.,1.,0.}; // normalization, pi/K ratio, deltaGamma
    paramFixSpec fixedParams;
    //============================== SETTINGS ===================================//
    //============================== SETTINGS ===================================//

    // read nusquids calculated flux
    nuSQUIDSAtm<nuSQUIDSLV> nus_kaon((std::string(argv[3])));
    nuSQUIDSAtm<nuSQUIDSLV> nus_pion((std::string(argv[4])));

    //============================= begin read data  =============================//
    marray<double,2> observed_data = quickread(std::string(argv[2]));
    std::deque<Event> observed_events;
    for(unsigned int irow = 0; irow < observed_data.extent(0); irow++){
      observed_events.push_back(Event(observed_data[irow][0],// energy proxy
                                      observed_data[irow][1],// costh
                                      static_cast<unsigned int>(observed_data[irow][2])));// year
    }
    //============================= end read data  =============================//

    //============================= begin calculating event expectation  =============================//

    const unsigned int neutrino_energy_index = 0;
    const unsigned int coszenith_index = 1;
    const unsigned int proxy_energy_index = 2;
    AreaEdges edges;
    AreaArray areas = get_areas(std::string(argv[1]), edges);

    //std::cout << "Printing out the number of edges. Just a sanity check." << std::endl;
    //std::cout << edges[y2010][NUMU][neutrino_energy_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][coszenith_index].extent(0) << std::endl;
    //std::cout << edges[y2010][NUMU][proxy_energy_index].extent(0) << std::endl;

    const unsigned int neutrinoEnergyBins=280;
    const unsigned int cosZenithBins=11;
    const unsigned int energyProxyBins=50;
    const unsigned int number_of_years = 2;
    const double pi = 3.141592;
    const double m2Tocm2 = 1.0e4;
    std::map<Year,double> livetime {{y2010,2.7282e+07},{y2011,2.96986e+07}}; // in seconds

    nusquids::marray<double,3> kaon_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    for(auto it = kaon_event_expectation.begin(); it < kaon_event_expectation.end(); it++)
        *it = 0.;

    nusquids::marray<double,3> pion_event_expectation{number_of_years,cosZenithBins,energyProxyBins};
    for(auto it = pion_event_expectation.begin(); it < pion_event_expectation.end(); it++)
        *it = 0.;

    for(Year year : {y2010,y2011}){
        for(PTypes flavor : {NUMU,NUMUBAR}){
            for(unsigned int ci = 0; ci < cosZenithBins-1; ci++){
                for(unsigned int pi = 0; pi < energyProxyBins-1; pi++){
                    for(unsigned int ei = 0; ei < neutrinoEnergyBins-1; ei++){
                        double solid_angle = 2.*pi*(edges[year][flavor][coszenith_index][ci+1]-edges[year][flavor][coszenith_index][ci]);
                        kaon_event_expectation[year][ci][pi] += solid_angle*m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*GetAveragedFlux(&nus_kaon,flavor,
                                                                                                                       edges[year][flavor][coszenith_index][ci],
                                                                                                                       edges[year][flavor][coszenith_index][ci+1],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei+1]);
                        pion_event_expectation[year][ci][pi] += solid_angle*m2Tocm2*livetime[year]*areas[year][flavor][ei][ci][pi]*GetAveragedFlux(&nus_pion,flavor,
                                                                                                                       edges[year][flavor][coszenith_index][ci],
                                                                                                                       edges[year][flavor][coszenith_index][ci+1],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei],
                                                                                                                       edges[year][flavor][neutrino_energy_index][ei+1]);
                    }
                }
            }
        }
    }

    // to keep the code simple we are going to construct fake MC events
    // which weights are just the expectation value. The advantage of this is that
    // when we have the actual IceCube MC the following code can be reused. Love, CA.

    std::deque<Event> mc_events;

    for(Year year : {y2010,y2011}){
      for(unsigned int ci = 0; ci < cosZenithBins-1; ci++){
        for(unsigned int pi = 0; pi < energyProxyBins-1; pi++){
          // we are going to save the bin content and label it at the bin center
          mc_events.push_back(Event((edges[year][flavor][proxy_energy_index][pi]+edges[year][flavor][proxy_energy_index][pi+1])/2., // energy proxy bin center
                                    (edges[year][flavor][coszenith_index][ci]+edges[year][flavor][coszenith_index][ci+1])/2., // costh bin center
                                    year == y2010 ? 2010 : 2011, // year
                                    kaon_event_expectation[year][ci][pi], // amount of kaon component events
                                    pion_event_expectation[year][ci][pi])); // amount of pion component events
        }
      }
    }

    //============================= end calculating event expectation  =============================//

    //============================= begin making histograms  =============================//
    // LLH problem and histogram
    // lets construct to weaver histograms
    using namespace phys_tools::histograms;
    using HistType = histogram<3,likelihood::entryStoringBin<std::reference_wrapper<const Event>>>;
    using phys_tools::histograms::amount;
    auto binner = [](HistType& h, const Event& e){
      h.add(e.energy_proxy,e.costh,e.year,amount(std::cref(e)));
    };
    // set up data histogram // note that the energy proxy and costh bins are the same for both years
    HistType data_hist(FixedUserDefinedAxis(edges[y2010][flavor][coszenith_index]),
                       FixedUserDefinedAxis(edges[y2011][flavor][proxy_energy_index]),
                       LinearAxis(2010,1));

    /*
    data_hist.getAxis(0)->SetLowerLimit(minFitEnergy);
    data_hist.getAxis(0)->SetUpperLimit(maxFitEnergy);
    data_hist.getAxis(1)->SetLowerLimit(minCosth);
    data_hist.getAxis(1)->SetUpperLimit(maxCosth);
    */

    // fill in the histogram with the data
    bin(observed_events,data_hist,binner);

    // create MC histogram with the same binning as the data
    HistType sim_hist = makeEmptyHistogramCopy(data_hist);
    // fill in the histogram with the mc events
    bin(mc_events,sim_hist,binner);

    //============================= end making histograms  =============================//

    //============================= likelihood problem begins  =============================//
    // priors on the nuisance parameters

    likelihood::UniformPrior positivePrior(0.0,std::numeric_limits<double>::infinity());
    likelihood::GaussianPrior normalizationPrior(1.,0.4);
    likelihood::GaussianPrior crSlopePrior(0.0,0.05);
    likelihood::GaussianPrior kaonPrior(1.0,0.1);

    auto priors=makePriorSet(normalizationPrior,crSlopePrior,kaonPrior);
    // construct a MC event reweighter
    DiffuseFitWeighterMaker DFWM;
    // construct likelihood problem
    // there are two numbers here. The first number is the number of histogram dimension
    // in this case 3. The second number is the number of nuisance parameters, also 3.
    auto prob=likelihood::makeLikelihoodProblem<std::reference_wrapper<const Event>,3,3>(data_hist, {sim_hist}, priors, {1.0}, likelihood::simpleDataWeighter(), DFWM, likelihood::poissonLikelihood(), fitSeed);
    prob.setEvaluationThreadCount(evalThreads);

    std::vector<double> seed=prob.getSeed();
    std::vector<unsigned int> fixedIndices;
    for(const auto pf : fixedParams.params){
      if(!quiet)
        std::cout << "Fitting with parameter " << pf.first << " fixed to " << pf.second << std::endl;
      seed[pf.first]=pf.second;
      fixedIndices.push_back(pf.first);
    }
    // minimize over the nuisance parameters
    fitResult fr = doFitLBFGSB(prob,seed,fixedIndices);

    if(!quiet)
      std::cout << "Fitted Hypothesis: ";
    for(unsigned int i=0; i<fr.params.size(); i++)
      std::cout << fr.params[i] << ' ';
    std::cout << std::setprecision(10) << fr.likelihood << std::setprecision(6) << ' ' << (fr.succeeded?"succeeded":"failed") << std::endl;


    //============================= likelihood problem ends =============================//

    std::string output_file_str;;
    if(argc > 5)
      output_file_str=std::string(argv[5]);
    else
      output_file_str=std::string("./expectation.dat");

    std::ofstream output_file(output_file_str);
    auto weighter = DFWM(fr.params);
    for(auto event : mc_events){
      output_file << event.energy_proxy << " " << event.costh << " " << event.year << " " << weighter(event) << std::endl;
    }
    output_file.close();

    return 0;
}
