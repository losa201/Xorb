export interface Company {
  id: string;
  name: string;
  logo: string;
  metadata: {
    order: number;
    isVisible: boolean;
  };
}

export interface CompanyModule {
  companies: Company[];
  metadata: {
    updatedAt: Date;
    version: number;
  };
}