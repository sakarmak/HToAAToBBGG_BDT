void BDTFileMaker(){
	float Leppt, Lepeta, Lepphi, Lepe;
	float B1pt, B1eta, B1phi, B1e;
	float B2pt, B2eta, B2phi, B2e;
	float BB_inv_mass, B1_DeepFlv, B2_DeepFlv;
	float Pho1pt, Pho1eta, Pho1phi, Pho1e, Pho1MVA;
	float Pho2pt, Pho2eta, Pho2phi, Pho2e, Pho2MVA;
	float Dipho_invmass, Invmassbbgg;
	double weight_nom;
	bool isEle, isMu;
	int ab, bb;

	float leppt, lepeta, lepphi, lepe;
        float b1pt, b1eta, b1phi, b1e;
        float b2pt, b2eta, b2phi, b2e;
        float bb_inv_mass, b1_DeepFlv, b2_DeepFlv;
        float pho1pt, pho1eta, pho1phi, pho1e, pho1MVA;
        float pho2pt, pho2eta, pho2phi, pho2e, pho2MVA;
        float m_gg, m_bbgg;
        float evt_wgt;
	int yout;
	

	TFile *fout = new TFile("TTSL_2018_Mu_twoB.root","RECREATE");
        TTree *tree = new TTree("tree","tree");

   	tree->Branch("leppt",&leppt);
	tree->Branch("lepeta",&lepeta);
	tree->Branch("lepphi",&lepphi);
	
	tree->Branch("b1pt",&b1pt);
        tree->Branch("b1eta",&b1eta);
        tree->Branch("b1phi",&b1phi);
	tree->Branch("b2pt",&b2pt);
        tree->Branch("b2eta",&b2eta);
        tree->Branch("b2phi",&b2phi);

	tree->Branch("pho1pt",&pho1pt);
        tree->Branch("pho1eta",&pho1eta);
        tree->Branch("pho1phi",&pho1phi);
        tree->Branch("pho2pt",&pho2pt);
        tree->Branch("pho2eta",&pho2eta);
        tree->Branch("pho2phi",&pho2phi);
	tree->Branch("m_gg",&m_gg);
	tree->Branch("m_bbgg",&m_bbgg);
	tree->Branch("evt_wgt",&evt_wgt);
	tree->Branch("yout",&yout);

	TFile *f = new TFile("TTSL_2018_V1.root");
        TTree *tr = (TTree*)f->Get("Tout");

        tr->SetBranchAddress("leppt",&Leppt);
        tr->SetBranchAddress("lepeta",&Lepeta);
        tr->SetBranchAddress("lepphi",&Lepphi);

        tr->SetBranchAddress("b1pt",&B1pt);
        tr->SetBranchAddress("b1eta",&B1eta);
        tr->SetBranchAddress("b1phi",&B1phi);
        tr->SetBranchAddress("b2pt",&B2pt);
        tr->SetBranchAddress("b2eta",&B2eta);
        tr->SetBranchAddress("b2phi",&B2phi);

        tr->SetBranchAddress("pho1pt",&Pho1pt);
        tr->SetBranchAddress("pho1eta",&Pho1eta);
        tr->SetBranchAddress("pho1phi",&Pho1phi);
        tr->SetBranchAddress("pho2pt",&Pho2pt);
        tr->SetBranchAddress("pho2eta",&Pho2eta);
        tr->SetBranchAddress("pho2phi",&Pho2phi);
        tr->SetBranchAddress("dipho_invmass",&Dipho_invmass);
        tr->SetBranchAddress("invmassbbgg",&Invmassbbgg);
	tr->SetBranchAddress("isMu",&isMu);
	tr->SetBranchAddress("isEle",&isEle);
	tr->SetBranchAddress("ab",&ab);
	tr->SetBranchAddress("bb",&bb);
	tr->SetBranchAddress("weight_nom",&weight_nom);


	double TTGJets= (59730*4.078)/(2.66557e+07);
	double DY= (59730*5343)/(9.14154e+07);
	double TTSL= (59730*365.34)/(1.48113e+11);
	double TT2L2Nu= (59730*88.29)/(1.03533e+10);
	double WH_mA20= (59730*0.01)/(993834);
	double WH_mA55= (59730*0.01)/(903853);

	for(int ij=0; ij<tr->GetEntries(); ++ij){
		tr->GetEntry(ij);
		if(!isMu) continue;
		if(bb==0) continue;
		leppt = Leppt,
		lepeta = Lepeta;
		lepphi = Lepphi;
		b1pt = B1pt;
		b1eta = B1eta;
		b1phi = B1phi;
		b2pt = B2pt;
                b2eta = B2eta;
                b2phi = B2phi;
		pho1pt = Pho1pt;
		pho1eta = Pho1eta;
		pho1phi = Pho1phi;
		pho2pt = Pho2pt;
                pho2eta = Pho2eta;
                pho2phi = Pho2phi;
		m_gg = Dipho_invmass;
		m_bbgg = Invmassbbgg;
		evt_wgt = weight_nom*TT2L2Nu;
		yout = 0;

		tree->Fill();
	}
	fout->cd();
	tree->Write();
	fout->Close();
}
