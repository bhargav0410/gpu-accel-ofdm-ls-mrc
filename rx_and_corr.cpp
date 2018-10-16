#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/transport/udp_simple.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp> //gets time
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <complex>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <ctime>
#include <sys/socket.h>
#include <string>
#include "ShMemSymBuff_gpu.hpp"

#define FFT_size dimension
#define numSymbols lenOfBuffer
#define mode 1
ShMemSymBuff* buffPtr;
int iter = 1;

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

std::vector<std::vector<std::complex<float> > > copy_buff;
int cp_size;

namespace po = boost::program_options;

void copy_to_shared_mem(int chan) {
	//std::cout << "Here\n";
	std::complex<float>* copy_to_mem = 0;
	copy_to_mem = (std::complex<float>*)malloc((chan*(FFT_size)*sizeof(*copy_to_mem)));
//	std::cout << "Num symbols: " << numSymbols << std::endl;
//	std::cout << "Prefix: " << cp_size << std::endl;
//	std::cout << "FFT size: " << FFT_size << std::endl;
	for (int i = 0; i < numSymbols; i++) {
//		std::cout << "Symbol: " << i+1 << std::endl;
		for (int j = 0; j < chan; j++) {
			memcpy(&copy_to_mem[j*(FFT_size)], &copy_buff[j][i*(FFT_size+cp_size) + cp_size], (FFT_size)*sizeof(*copy_to_mem));
			//memcpy(&copy_to_mem[j*(FFT_size+cp_size)], &copy_buff[j][i*(FFT_size+cp_size)], (FFT_size+cp_size)*sizeof(*copy_to_mem));
			/*
			for (int k = 0; k < FFT_size+cp_size; k++) {
				copy_to_mem[j*(FFT_size+cp_size) + k].real = copy_buff[j][i*(FFT_size+cp_size) + k].real();
				copy_to_mem[j*(FFT_size+cp_size) + k].imag = copy_buff[j][i*(FFT_size+cp_size) + k].imag();
			}
			*/
		}
		buffPtr->writeNextSymbolNoWait(copy_to_mem);
	}
	iter++;
	free(copy_to_mem);
}

int UHD_SAFE_MAIN(int argc, char *argv[]){
    uhd::set_thread_priority_safe();
	
	int ctr = 0;
    //variables to be set by po
    std::string args, ant, subdev, otw, ref, file, channel_list, sync;
    size_t num_samps;
    double rate, freq, gain, bw;
    float delay, thres;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
		("file-prefix", po::value<std::string>(&file)->default_value("corr_rec"), "prefix of the file name to write binary samples to")
        // hardware parameters
        ("rate", po::value<double>(&rate), "rate of incoming samples (sps)")
        ("freq", po::value<double>(&freq), "RF center frequency in Hz")
        ("gain", po::value<double>(&gain), "gain for the RF chain")
        ("ant", po::value<std::string>(&ant), "antenna selection")
        ("subdev", po::value<std::string>(&subdev), "subdevice specification")
        ("bw", po::value<double>(&bw), "analog frontend filter bandwidth in Hz")
		("otw", po::value<std::string>(&otw)->default_value("sc16"), "specify the over-the-wire sample mode")
		("channels", po::value<std::string>(&channel_list)->default_value("0"), "which channel(s) to use (specify \"0\", \"1\", \"0,1\", etc)")
        ("sync", po::value<std::string>(&sync)->default_value("now"), "synchronization method: now, pps, mimo")
        ("frame-size", po::value<size_t>(&num_samps)->default_value(1024), "number of samples per buffer")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
        ("int-n", "tune USRP with integer-N tuning")
		("delay", po::value<float>(&delay)->default_value(0), "initial receive delay")
		("thres", po::value<float>(&thres)->default_value(0.1), "correlator threshold (peaks below this threshold will not be considered)")
		("cp-size", po::value<int>(&cp_size)->default_value(72), "size of cyclic prefix")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help") or not vm.count("rate")){
        std::cout << boost::format("UHD RX to file with sync %s") % desc << std::endl;
        return EXIT_FAILURE;
    }

    //create a usrp device
    std::cout << std::endl;
    std::cout << boost::format("Creating the usrp device with: %s...") % args << std::endl;
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);

    //Lock mboard clocks
    usrp->set_clock_source(ref);

     //always select the subdevice first, the channel mapping affects the other settings
    if (vm.count("subdev")) usrp->set_rx_subdev_spec(subdev);

    std::cout << boost::format("Using Device: %s") % usrp->get_pp_string() << std::endl;

	//detect which channels to use
    std::vector<std::string> channel_strings;
    std::vector<size_t> channel_nums;
    boost::split(channel_strings, channel_list, boost::is_any_of("\"',"));
    for(size_t ch = 0; ch < channel_strings.size(); ch++){
        size_t chan = boost::lexical_cast<int>(channel_strings[ch]);
        if(chan >= usrp->get_rx_num_channels()){
            throw std::runtime_error("Invalid channel(s) specified.");
        }
        else channel_nums.push_back(boost::lexical_cast<int>(channel_strings[ch]));
    }
	
    //set the sample rate
    if (not vm.count("rate")){
        std::cerr << "Please specify the sample rate with --rate" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rate/1e6) << std::endl;
    usrp->set_rx_rate(rate);
    std::cout << boost::format("Actual RX Rate: %f Msps...") % (usrp->get_rx_rate()/1e6) << std::endl << std::endl;

	for (int ch = 0; ch < channel_nums.size(); ch++) {
		//set the center frequency
		if (not vm.count("freq")){
			std::cerr << "Please specify the center frequency with --freq" << std::endl;
			return EXIT_FAILURE;
		}
		
		uhd::tune_request_t tune_request(freq, ch);
		if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
		usrp->set_rx_freq(tune_request, ch);
		std::cout << boost::format("Actual RX Freq: %f MHz...") % (usrp->get_rx_freq()/1e6) << std::endl << std::endl;

		//set the rf gain
		if (vm.count("gain")){
			std::cout << boost::format("Setting RX Gain: %f dB...") % gain << std::endl;
			usrp->set_rx_gain(gain, ch);
			std::cout << boost::format("Actual RX Gain: %f dB...") % usrp->get_rx_gain() << std::endl << std::endl;
		}

		//set the analog frontend filter bandwidth
		if (vm.count("bw")){
			std::cout << boost::format("Setting RX Bandwidth: %f MHz...") % (bw/1e6) << std::endl;
			usrp->set_rx_bandwidth(bw, ch);
			std::cout << boost::format("Actual RX Bandwidth: %f MHz...") % (usrp->get_rx_bandwidth()/1e6) << std::endl << std::endl;
		}

		//set the antenna
		if (vm.count("ant")) {
			std::cout << boost::format("Setting RX Antenna: %f...") % (ant) << std::endl;
			usrp->set_rx_antenna(ant, ch);
			std::cout << boost::format("Actual RX Antenna: %f...") % (usrp->get_rx_antenna(ch)) << std::endl << std::endl;
		}
	}
	
	
    boost::this_thread::sleep(boost::posix_time::seconds(1)); //allow for some setup time

    //Check Ref and LO Lock detect
	/*
    std::vector<std::string> sensor_names;
    sensor_names = usrp->get_rx_sensor_names(0);
    if (std::find(sensor_names.begin(), sensor_names.end(), "lo_locked") != sensor_names.end()) {
        uhd::sensor_value_t lo_locked = usrp->get_rx_sensor("lo_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    sensor_names = usrp->get_mboard_sensor_names(0);
    if ((ref == "mimo") and (std::find(sensor_names.begin(), sensor_names.end(), "mimo_locked") != sensor_names.end())) {
        uhd::sensor_value_t mimo_locked = usrp->get_mboard_sensor("mimo_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % mimo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(mimo_locked.to_bool());
    }
    if ((ref == "external") and (std::find(sensor_names.begin(), sensor_names.end(), "ref_locked") != sensor_names.end())) {
        uhd::sensor_value_t ref_locked = usrp->get_mboard_sensor("ref_locked",0);
        std::cout << boost::format("Checking RX: %s ...") % ref_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(ref_locked.to_bool());
    }
	*/
	
	std::ofstream outfile;
	std::string outfilename;
	std::ifstream infile;
	std::string infilename = "PNSeq_255_MaxLenSeq.dat";
	infile.open(infilename.c_str(), std::ifstream::binary);
	infile.seekg(0, infile.end);
	size_t num_tx_samps = infile.tellg()/sizeof(std::complex<float>);
	infile.seekg(0, infile.beg);
	std::vector<std::complex<float> > pn_buff(num_tx_samps);
	infile.read((char*)&pn_buff.front(), num_tx_samps*sizeof(std::complex<float>));
	std::cout << "PN length: " << pn_buff.size() << "\n\n";
	
	std::cout << boost::format("Setting device timestamp to 0...") << std::endl;
    if (sync == "now"){
        //This is not a true time lock, the devices will be off by a few RTT.
        //Rather, this is just to allow for demonstration of the code below.
        usrp->set_time_now(uhd::time_spec_t(0.0));
    }
    else if (sync == "pps"){
        usrp->set_time_source("external");
        usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));
        boost::this_thread::sleep(boost::posix_time::seconds(1)); //wait for pps sync pulse
    }
    else if (sync == "mimo"){
        UHD_ASSERT_THROW(usrp->get_num_mboards() == 2);

        //make mboard 1 a slave over the MIMO Cable
        usrp->set_clock_source("mimo", 1);
        usrp->set_time_source("mimo", 1);

        //set time on the master (mboard 0)
        usrp->set_time_now(uhd::time_spec_t(0.0), 0);

        //sleep a bit while the slave locks its time to the master
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    }
	
	
	
    //create a receive streamer
    std::vector<std::vector<std::complex<float> > > buff1(
        channel_nums.size(), std::vector<std::complex<float> >(num_samps)
    );
	std::vector<std::vector<std::complex<float> > > buff2(
        channel_nums.size(), std::vector<std::complex<float> >(num_samps)
    );
	
	
	copy_buff.resize(
        channel_nums.size(), std::vector<std::complex<float> >(num_samps - pn_buff.size())
    );
	
	
	std::vector<std::complex<float> *> buff_ptrs1;
    for (size_t i = 0; i < buff1.size(); i++) buff_ptrs1.push_back(&buff1[i].front());
	std::vector<std::complex<float> *> buff_ptrs2;
    for (size_t i = 0; i < buff2.size(); i++) buff_ptrs2.push_back(&buff2[i].front());
	
	uhd::stream_args_t stream_args("fc32",otw);
	stream_args.channels = channel_nums;
	uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);
	uhd::rx_metadata_t md;
	
	uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
	stream_cmd.stream_now = false;
	stream_cmd.time_spec = uhd::time_spec_t(delay);
	double timeout = delay + 0.5;
	rx_stream->issue_stream_cmd(stream_cmd);
    //------------------------------------------------------------------
    //-- Initialize
    //------------------------------------------------------------------
	std::signal(SIGINT, &sig_int_handler);
	
	
	int samps_per_buff = num_samps, length = 0;//, iter = 0;
	bool corr_flag = false, copy_flag = false, first_time = true;
	std::string shm_uid = shmemID;
	buffPtr = new ShMemSymBuff(shm_uid, mode);
	boost::thread t;
	size_t num_rx_samps;
	while (not stop_signal_called){
		//corr_flag = false;
		num_rx_samps = rx_stream->recv(buff_ptrs1, samps_per_buff, md, timeout);
		stream_cmd.stream_now = true;
		timeout = 0.5;
		num_rx_samps = rx_stream->recv(buff_ptrs2, samps_per_buff, md, timeout);
		//std::cout << num_rx_samps << std::endl;
		
		/*
		if (first_time == false) {
			for (int ch = 0; ch < channel_nums.size(); ch++) {
				int j = 0;
				for (int i = length + pn_buff.size(); i < samps_per_buff; i++) {
					copy_buff[ch][j] = buff[ch][i];
					j++;
				}
			}
			copy_to_shared_mem(copy_buff, channel_nums.size());
			break;
		}
		*/
		
		
		
		std::vector<std::complex<float> > temp(num_rx_samps);
		std::complex<float> abs;
		float temp_iter;
		if (corr_flag == false) {
			for (int ch_temp = 0; ch_temp < channel_nums.size(); ch_temp++) {
				for (int i = 0; i < (samps_per_buff - pn_buff.size() + 1); i++) {
					temp[i] = 0;
					for (int j = 0; j < pn_buff.size(); j++) {
						/*
						float tempr = pn_buff[j].real()*buff[0][i+j].real() - pn_buff[j].imag()*buff[0][i+j].imag();
						float tempi = pn_buff[j].real()*buff[0][i+j].imag() + pn_buff[j].imag()*buff[0][i+j].real();
						abs.real(tempr);
						abs.imag(tempi);
						temp[i] += std::sqrt(abs.real()*abs.real() + abs.imag()*abs.imag());
						*/
						temp[i] += pn_buff[j]*buff1[ch_temp][i+j];
						//temp_iter = i;
					}
					temp_iter = std::abs(temp[i])/((float)pn_buff.size());
					if (temp_iter >= thres) {
						std::cout << "\n" << temp_iter << "\n";
						std::cout << "\n" << i << "\n";
						length = i;
						corr_flag = true;
						break;
					}
				}
				if (temp_iter >= thres) {
					break;
				}
			}
		}
		
		if (temp_iter < thres) {
			continue;
		}
		
		if (corr_flag == true and first_time == false)
			{t.join();}
		else if (corr_flag == true and first_time == true)
			{first_time = false;}
		
		
		if (corr_flag == true) { // and first_time == true) {
			for (int ch = 0; ch < channel_nums.size(); ch++) {
				int j = 0;
				for (int i = length + pn_buff.size(); i < samps_per_buff; i++) {
					copy_buff[ch][j] = buff1[ch][i];
					j++;
				}
			}
		}


		if (length > 0 and corr_flag == true) {// and first_time == true) {
			for (int ch = 0; ch < channel_nums.size(); ch++) {
				for (int i = 0; i < length; i++) {
					copy_buff[ch][i + (samps_per_buff - length - pn_buff.size())] = buff2[ch][i];
				}
			}
			//length = 0;
			//break;
			
			copy_flag = true;
		}
		
		
		if (copy_flag == true) {// and first_time == true) {
			//Shared memory part
			int ch = channel_nums.size();
			t = boost::thread(copy_to_shared_mem, boost::ref(ch));
			//t.join();
			//iter++;
			//if (iter > 1) break;
			//first_time = false;
			//continue;
		}
		
    }
	rx_stream->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
	t.join();
	
	if (not file.empty()) {
		for (int i = 0; i < channel_nums.size(); i++) {
			outfilename = file + "_ch_" + boost::lexical_cast<std::string>(i) + "_binary";
			outfile.open(outfilename.c_str(), std::ofstream::binary);
			outfile.write((const char*)&copy_buff[i].front(), copy_buff[i].size()*sizeof(std::complex<float>));
			if (outfile.is_open()) {
				outfile.close();
			}
			outfilename = file + "_ch_dumped_" + boost::lexical_cast<std::string>(i) + "_binary";
			outfile.open(outfilename.c_str(), std::ofstream::binary);
			outfile.write((const char*)&buff1[i].front(), buff1[i].size()*sizeof(std::complex<float>));
			outfile.write((const char*)&buff2[i].front(), buff2[i].size()*sizeof(std::complex<float>));
			if (outfile.is_open()) {
				outfile.close();
			}
		}
	}
	
    //------------------------------------------------------------------
    //-- Cleanup
    //------------------------------------------------------------------
    //finished
	delete buffPtr;
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return EXIT_SUCCESS;
}