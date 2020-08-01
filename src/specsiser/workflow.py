






# def inference_model_selection(objProperties, objLinesDF, ion_model, extinction_model, chemistry_model):
#
#     # Compute extinction curve for the observed emission lines
#     lineFlambdas = extinction_model.gasExtincParams(wave=objLinesDF.obsWave.values)
#
#     # Tag the emission features for the chemical model implementation
#     chemistry_obj = chemistry_model()
#     chemistry_obj.label_ion_features(linesDF=objLinesDF, highTempIons=objProperties['high_temp_ions_list'])
#
#
#     return
#
# class ModelIngredients(SspFitter, NebularContinuaCalculator, EmissionComponents, ReddeningLaws):
#
#     def prepareSimulation(self, obs_data, ssp_data=None, output_folder = None, storage_folder = None,
#                         spectra_components=None, input_lines='all', normalized_by_Hbeta=True,
#                         excludeReddening = False, T_high_prior = False, prefit_ssp = True,
#                         wavelengh_limits = None, resample_inc = None, norm_interval=None):
#
#         # Store components fit
#         self.spectraComponents = spectra_components
#
#         # Folders to store inputs and outputs
#         self.input_folder       = output_folder + 'input_data/' # TODO sure?
#         self.output_folder      = output_folder + 'output_data/'
#         self.dataFolder         = self.output_folder if storage_folder is None else storage_folder # TODO sure?
#         self.configFile         = obs_data['obsFile'] #TODO this one has to go into the configuration
#         self.objName            = str(obs_data['objName'])
#         self.prefit_db          = '{}{}_sspPrefitDB'.format(self.dataFolder, self.objName)
#         self.sspCoeffsPrefit_file = '{}{}_prefitSSPpopulations.txt'.format(self.input_folder, self.objName)
#         self.sspCoeffs_file     = '{}{}_SSPpopulations.txt'.format(self.input_folder, self.objName)
#
#         # Create them if not available
#         make_folder(self.input_folder)
#         make_folder(self.output_folder)
#
#         # Prepare spectrum components for fitting
#         self.emissionCheck, self.stellarCheck, self.emissionCheck = False, False, False
#
#         # Pre-analysis emission spectrum
#         if 'emission' in self.spectraComponents:
#
#             # Get emission data from input files
#             self.ready_simulation(output_folder, obs_data, ssp_data, spectra_components, input_lines=input_lines,
#                                   wavelengh_limits=wavelengh_limits, resample_inc=resample_inc, norm_interval=norm_interval)
#
#             # Declare gas sampler variables
#             self.gasSamplerVariables(self.obj_data['lineIons'], self.config['high_temp_ions'],
#                                      self.obj_data['lineFluxes'], self.obj_data['lineErr'],
#                                      self.obj_data['lineLabels'], self.obj_data['lineFlambda'],
#                                      normalized_by_Hbeta, self.config['linesMinimumError'])
#
#             # Prios definition # TODO this must go to the a special section
#             self.priorsDict = {'T_low': self.obj_data['Te_prior'], 'n_e': self.obj_data['ne_prior']}
#
#             # Confirm inputs are valid
#             self.emissionCheck = True
#
#         # Prefit stellar continua
#         if 'stellar' in self.spectraComponents:
#
#             self.stellarCheck = True
#
#             # Perform a new SPP synthesis otherwise use available data
#             if prefit_ssp:
#
#                 # Compute nebular continuum using normalise Halpha and standard conditions
#                 self.computeDefaultNeb(self.nebDefault['Te_neb'], self.obj_data['nebFlambda'], self.nebDefault['cHbeta_neb'],
#                                        self.nebDefault['He1_neb'], self.nebDefault['He2_neb'],
#                                        self.nebDefault['flux_halpha'] / self.obj_data['normFlux_coeff'], self.nebDefault['z_neb'])
#
#                 # Ready continuum data
#                 self.prepareContinuaData(self.ssp_lib['wave_resam'], self.ssp_lib['flux_norm'], self.ssp_lib['normFlux_coeff'],
#                                          self.obj_data['wave_resam'], self.obj_data['flux_norm'], self.obj_data['continuum_sigma'],
#                                          self.int_mask, nebularFlux=self.nebDefault['synth_neb_flux'])
#
#                 # Select model
#                 self.select_inference_model('stelar_prefit')
#
#                 # Plot input simulation data
#                 self.plotInputSSPsynthesis()
#
#                 # Run stellar continua prefit and store/print the results
#                 #self.run_pymc(self.prefit_db, iterations=8000, variables_list=['Av_star', 'sigma_star'], prefit = True)
#                 self.savePrefitData(self.sspCoeffsPrefit_file, self.prefit_db)
#
#             # Compute nebular continuum using prior physical data
#             self.computeDefaultNeb(self.nebDefault['Te_neb'], self.obj_data['nebFlambda'], self.nebDefault['cHbeta_neb'],
#                                    self.nebDefault['He1_neb'], self.nebDefault['He2_neb'],
#                                    self.obj_data['flux_halpha'] / self.obj_data['normFlux_coeff'], self.nebDefault['z_neb'])
#
#             # Compute nebular continuum using normalise Halpha and standard conditions
#             # TODO I think I need to remove nebular continuum here if I want to add it later
#             self.prepareContinuaData(self.ssp_lib['wave_resam'], self.ssp_lib['flux_norm'], self.ssp_lib['normFlux_coeff'],
#                                      self.obj_data['wave_resam'], self.obj_data['flux_norm'],
#                                      self.obj_data['continuum_sigma'],
#                                      self.int_mask,
#                                      nebularFlux=None,#self.nebDefault['synth_neb_flux'],
#                                      mainPopulationsFile=self.sspCoeffsPrefit_file)
#
#         return