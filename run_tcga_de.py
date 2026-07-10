#!/usr/bin/env python3
"""Matched TCGA PanCancer Atlas GBM mutation/expression differential analysis."""
from __future__ import annotations

import json, math, re, tarfile, traceback, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT=Path.cwd(); DATA=ROOT/'downloaded_data'; OUT=ROOT/'results'
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
STUDY='gbm_tcga_pan_can_atlas_2018'
TARBALL_URLS=[
    f'https://datahub.assets.cbioportal.org/{STUDY}.tar.gz',
    f'https://cbioportal-datahub.s3.amazonaws.com/{STUDY}.tar.gz',
]
FOCUSED=['SF3B1','SRSF2','U2AF1','ZRSR2','PRPF8','LUC7L2','RBM10','FUBP1','QKI','HNRNPK','DDX41','PCBP1','RBM5','RBM6','SF1','U2AF2','SRSF1','TRA2B']
ADDITIONS=['PTBP1','SRRM2','RBM25','RBM47','HNRNPU','HNRNPL','HNRNPA2B1']
EXPANDED=FOCUSED+ADDITIONS
CANONICAL15=['SF3B1','SF3A1','U2AF1','U2AF2','SRSF2','ZRSR2','RBM10','PRPF8','LUC7L2','DDX3X','SUGP1','PHF5A','RBM39','HNRNPK','FUBP1']
PANELS={'expanded_explicit25':EXPANDED,'focused_explicit18':FOCUSED,'canonical15_sensitivity':CANONICAL15}
QUALIFYING={'Missense_Mutation','Nonsense_Mutation','Frame_Shift_Del','Frame_Shift_Ins','In_Frame_Del','In_Frame_Ins','Splice_Site','Translation_Start_Site','Nonstop_Mutation','Start_Codon_Del','Start_Codon_Ins','De_novo_Start_OutOfFrame','De_novo_Start_InFrame','Stop_Codon_Del','Stop_Codon_Ins'}


def download_tarball():
    dest=DATA/f'{STUDY}.tar.gz'; last=None
    for url in TARBALL_URLS:
        try:
            print('Downloading study package:',url,flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=600) as r,open(dest,'wb') as w:
                while True:
                    b=r.read(1024*1024)
                    if not b: break
                    w.write(b)
            print('Saved',dest,dest.stat().st_size,flush=True)
            if dest.stat().st_size<10000: raise RuntimeError('study package too small')
            return dest,url
        except Exception as e:
            print(' failed:',repr(e),flush=True); last=e
            if dest.exists(): dest.unlink()
    raise RuntimeError(f'Could not download study package: {last}')


def extract_study(path):
    out=DATA/'study'; out.mkdir(exist_ok=True)
    with tarfile.open(path,'r:gz') as tf: tf.extractall(out)
    files={p.name:p for p in out.rglob('*') if p.is_file()}
    need=['data_mrna_seq_v2_rsem.txt','data_mutations.txt']
    missing=[x for x in need if x not in files]
    if missing: raise RuntimeError(f'Missing study files: {missing}; available={sorted(files)[:50]}')
    return files


def patient_id(s):
    s=str(s); return s[:12] if s.startswith('TCGA-') and len(s)>=12 else s

def sample_type(s):
    m=re.match(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-([0-9]{2})',str(s),re.I)
    return m.group(1) if m else None


def read_expression(path):
    x=pd.read_csv(path,sep='\t',low_memory=False)
    gene_col=next((c for c in ['Hugo_Symbol','HUGO_SYMBOL','gene','Gene'] if c in x.columns),x.columns[0])
    meta=[c for c in [gene_col,'Entrez_Gene_Id','ENTREZ_GENE_ID'] if c in x.columns]
    samples=[c for c in x.columns if c not in meta]
    x=x[[gene_col]+samples].copy(); x[gene_col]=x[gene_col].astype(str)
    x=x.set_index(gene_col).apply(pd.to_numeric,errors='coerce').groupby(level=0).mean()
    raw_values=x.to_numpy(dtype=float)
    raw_min=float(np.nanmin(raw_values)); raw_max=float(np.nanmax(raw_values)); negative_count=int(np.sum(raw_values<0))
    if negative_count:
        raise RuntimeError(f'RSEM matrix contains {negative_count} negative values; cannot apply log2(RSEM+1)')
    print(f'Raw RSEM range: {raw_min:.6g} to {raw_max:.6g}; applying log2(RSEM+1)',flush=True)
    x=np.log2(x.clip(lower=0)+1.0)
    choices=[]
    for s in samples:
        code=sample_type(s)
        if code in (None,'01'): choices.append((patient_id(s),s,code or 'patient_level'))
    d=pd.DataFrame(choices,columns=['patient_id','expr_sample_id','sample_type_code']).sort_values(['patient_id','expr_sample_id']).drop_duplicates('patient_id')
    if d.empty: raise RuntimeError('No primary expression samples')
    return x,d.reset_index(drop=True)


def read_mutations(path, patients):
    hdr=pd.read_csv(path,sep='\t',nrows=0,comment='#')
    cols=hdr.columns.tolist()
    gcol=next(c for c in ['Hugo_Symbol','HUGO_SYMBOL'] if c in cols)
    scol=next(c for c in ['Tumor_Sample_Barcode','TUMOR_SAMPLE_BARCODE'] if c in cols)
    ccol=next(c for c in ['Variant_Classification','VARIANT_CLASSIFICATION'] if c in cols)
    wanted=set(patients); records=[]
    for chunk in pd.read_csv(path,sep='\t',comment='#',usecols=[gcol,scol,ccol],chunksize=100000,low_memory=False):
        chunk['patient_id']=chunk[scol].astype(str).map(patient_id)
        chunk=chunk[chunk['patient_id'].isin(wanted)]
        chunk=chunk[chunk[ccol].isin(QUALIFYING)]
        chunk=chunk[chunk[scol].astype(str).map(lambda z: sample_type(z) in (None,'01'))]
        if not chunk.empty:
            chunk=chunk.rename(columns={gcol:'gene',scol:'mutation_sample_id',ccol:'variant_classification'})
            records.append(chunk[['patient_id','mutation_sample_id','gene','variant_classification']])
    m=pd.concat(records,ignore_index=True) if records else pd.DataFrame(columns=['patient_id','mutation_sample_id','gene','variant_classification'])
    m['gene']=m['gene'].astype(str)
    return m.drop_duplicates()


def bh(p):
    p=np.asarray(p,float); q=np.full(p.shape,np.nan); ok=np.isfinite(p)
    if not ok.any(): return q
    v=p[ok]; order=np.argsort(v); r=v[order]; n=len(r)
    z=np.minimum.accumulate((r*n/np.arange(1,n+1))[::-1])[::-1]; inv=np.empty(n,int); inv[order]=np.arange(n)
    q[ok]=np.clip(z[inv],0,1); return q


def hg(x,y):
    nx,ny=len(x),len(y)
    if nx<2 or ny<2:return np.nan
    vx,vy=np.var(x,ddof=1),np.var(y,ddof=1); pv=((nx-1)*vx+(ny-1)*vy)/max(nx+ny-2,1)
    if pv<=0:return 0.0
    return (np.mean(x)-np.mean(y))/math.sqrt(pv)*(1-3/max(4*(nx+ny)-9,1))


def de(expr,group):
    gm=group.index[group]; gw=group.index[~group]; rows=[]
    for gene,row in expr.iterrows():
        a=pd.to_numeric(row[gm],errors='coerce').dropna().to_numpy(float); b=pd.to_numeric(row[gw],errors='coerce').dropna().to_numpy(float)
        z=np.r_[a,b]; ef=float(np.mean(z>0.1)) if len(z) else np.nan; sd=float(np.std(z,ddof=1)) if len(z)>1 else np.nan
        tested=len(a)>=3 and len(b)>=3 and np.isfinite(sd) and sd>0.1 and ef>=.20
        if tested:
            tt=stats.ttest_ind(a,b,equal_var=False,nan_policy='omit'); pw=float(tt.pvalue); tv=float(tt.statistic)
            try: pm=float(stats.mannwhitneyu(a,b,alternative='two-sided').pvalue)
            except Exception: pm=np.nan
        else: pw=tv=pm=np.nan
        rows.append({'gene_symbol':str(gene),'n_mut':len(a),'n_wt':len(b),'mean_log2_expr_mut':float(np.mean(a)) if len(a) else np.nan,'mean_log2_expr_wt':float(np.mean(b)) if len(b) else np.nan,'median_log2_expr_mut':float(np.median(a)) if len(a) else np.nan,'median_log2_expr_wt':float(np.median(b)) if len(b) else np.nan,'log2_fold_change':float(np.mean(a)-np.mean(b)) if len(a) and len(b) else np.nan,'hedges_g':hg(a,b),'welch_t':tv,'p_value_welch':pw,'p_value_mann_whitney':pm,'expressed_fraction':ef,'sd_all':sd,'tested':tested})
    r=pd.DataFrame(rows); r['fdr_bh']=bh(r.p_value_welch); r['significant_fdr05']=(r.fdr_bh<.05)&(r.log2_fold_change.abs()>=.5); r['nominal_p05']=(r.p_value_welch<.05)&(r.log2_fold_change.abs()>=.5); r['rank_score']=np.sign(r.log2_fold_change)*-np.log10(r.p_value_welch.clip(lower=1e-300))
    return r.sort_values(['fdr_bh','p_value_welch','gene_symbol'],na_position='last').reset_index(drop=True)


def volcano(r,title,path):
    d=r[r.tested & r.p_value_welch.notna()]
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(d.log2_fold_change,-np.log10(d.p_value_welch.clip(lower=1e-300)),s=9,alpha=.55)
    ax.axvline(-.5,ls='--',lw=.8); ax.axvline(.5,ls='--',lw=.8); ax.axhline(-math.log10(.05),ls='--',lw=.8); ax.set(xlabel='Mean difference on log2(RSEM + 1) scale',ylabel='-log10 Welch p-value',title=title)
    for _,x in d.nsmallest(12,'p_value_welch').iterrows(): ax.annotate(x.gene_symbol,(x.log2_fold_change,-math.log10(max(x.p_value_welch,1e-300))),fontsize=7)
    fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def pca(expr,group,path):
    z=expr.loc[expr.var(axis=1).nlargest(min(2000,len(expr))).index].T; a=z.to_numpy(float); a=np.nan_to_num(a-np.nanmean(a,axis=0),nan=0)
    u,s,_=np.linalg.svd(a,full_matrices=False); sc=u[:,:2]*s[:2]; ve=s*s/(s*s).sum(); fig,ax=plt.subplots(figsize=(7,6))
    for st,mk in [(False,'o'),(True,'^')]:
        mask=group.loc[z.index].to_numpy(bool)==st; ax.scatter(sc[mask,0],sc[mask,1],s=35,alpha=.7,marker=mk,label='Panel WT' if not st else 'Panel mutated')
    ax.set(xlabel=f'PC1 ({100*ve[0]:.1f}%)',ylabel=f'PC2 ({100*ve[1]:.1f}%)',title='TCGA IDH-wild-type GBM RNA-seq PCA'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def main():
    tar,url=download_tarball(); files=extract_study(tar); expr_raw,emap=read_expression(files['data_mrna_seq_v2_rsem.txt']); muts=read_mutations(files['data_mutations.txt'],emap.patient_id)
    mut_patients=set(muts.patient_id); emap=emap[emap.patient_id.isin(mut_patients)].sort_values('patient_id'); patients=emap.patient_id.tolist(); expr=pd.DataFrame({r.patient_id:expr_raw[r.expr_sample_id] for _,r in emap.iterrows()})
    idhmut=set(muts.loc[muts.gene.isin(['IDH1','IDH2']),'patient_id']); wt=[p for p in patients if p not in idhmut]; expr=expr[wt]; muts=muts[muts.patient_id.isin(wt)].copy()
    burden=muts.groupby('patient_id').gene.nunique().reindex(wt,fill_value=0).astype(int); sm=emap.set_index('patient_id').loc[wt].reset_index(); sm['nonsilent_mutated_gene_count']=sm.patient_id.map(burden); sm['idh1_or_idh2_nonsilent_mutation']=False
    expr.to_csv(OUT/'expression_log2_rsem_idhwt.csv.gz',compression='gzip')
    muts.to_csv(OUT/'nonsilent_mutations_idhwt.csv.gz',index=False,compression='gzip')
    emap.set_index('patient_id').loc[wt].reset_index().to_csv(OUT/'matched_expression_sample_map_idhwt.csv',index=False)
    summary={'data_provenance':{'study_id':STUDY,'study_tarball_url':url},'expression_file':'data_mrna_seq_v2_rsem.txt','mutation_file':'data_mutations.txt','expression_scale':'log2(cBioPortal TCGA PanCancer Atlas batch-normalized RSEM + 1)','mutation_definition':'Non-silent coding/splice-site classes from data_mutations.txt','n_primary_expression_patients':len(emap),'n_operational_idh_wildtype':len(wt),'n_operational_idh_mutant_excluded':len(idhmut & set(patients)),'idh_mutant_patient_ids':sorted(idhmut & set(patients)),'panels':{}}
    events=[]
    for label,panel in PANELS.items():
        ep=muts[muts.gene.isin(panel)][['patient_id','gene','mutation_sample_id','variant_classification']].drop_duplicates(); gp=pd.Series(False,index=wt); gp.loc[ep.patient_id.unique()]=True; sm['group_'+label]=sm.patient_id.map(gp).fillna(False); per=ep.groupby('patient_id').gene.apply(lambda x:';'.join(sorted(set(x)))); sm['mutated_panel_genes_'+label]=sm.patient_id.map(per).fillna(''); events.append(ep.assign(analysis=label))
        info={'genes_requested':panel,'n_mutated':int(gp.sum()),'n_wildtype':int((~gp).sum()),'mutated_patient_ids':sorted(gp.index[gp])}
        if gp.sum()>=3 and (~gp).sum()>=3:
            r=de(expr,gp); r.to_csv(OUT/f'de_{label}_all.csv',index=False); r.head(1000).to_csv(OUT/f'de_{label}_top1000.csv',index=False); r[r.significant_fdr05].to_csv(OUT/f'de_{label}_fdr05_fc05.csv',index=False); r[r.nominal_p05].to_csv(OUT/f'de_{label}_nominal_p05_fc05.csv',index=False); volcano(r,label,OUT/f'volcano_{label}.png')
            if label=='expanded_explicit25': pca(expr,gp,OUT/'pca_expanded_explicit25.png')
            info.update(n_genes_tested=int(r.tested.sum()),n_fdr05_and_abs_log2fc_ge_0_5=int(r.significant_fdr05.sum()),n_nominal_p05_and_abs_log2fc_ge_0_5=int(r.nominal_p05.sum()),top20=r.head(20)[['gene_symbol','log2_fold_change','p_value_welch','fdr_bh']].to_dict('records'))
        else: info['analysis_skipped_reason']='fewer than 3 samples in a group'
        summary['panels'][label]=info
    q95=float(burden.quantile(.95)); sm['hypermutation_q95_flag']=sm.nonsilent_mutated_gene_count>q95; summary['mutation_burden_q95']=q95; summary['hypermutation_q95_patient_ids']=sm.loc[sm.hypermutation_q95_flag,'patient_id'].tolist()
    sm.to_csv(OUT/'cohort_sample_metadata.csv',index=False); pd.concat(events,ignore_index=True).to_csv(OUT/'panel_mutation_events.csv',index=False); pd.DataFrame([{'analysis':k,'panel_gene':g,'panel_membership':'focused' if g in FOCUSED else ('expanded_addition' if g in ADDITIONS else 'canonical15_only')} for k,v in PANELS.items() for g in v]).to_csv(OUT/'panel_definitions.csv',index=False)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)); (OUT/'README.txt').write_text('Matched TCGA PanCancer Atlas GBM differential expression. Primary panel uses all 25 explicitly named genes from the prior report (18 focused + 7 additions). Non-silent mutations only. Expression was transformed as log2(RSEM + 1). Welch test plus BH FDR.\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__':
    try: main()
    except Exception:
        OUT.mkdir(exist_ok=True); (OUT/'failure_traceback.txt').write_text(traceback.format_exc()); traceback.print_exc(); raise
